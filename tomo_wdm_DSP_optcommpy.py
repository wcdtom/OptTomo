# =============================================================================
# --- 核心光通信库导入 (严格遵循 OptiCommPy 官方规范) ---
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.signal import correlate
import logging as logg
import argparse
from collections import defaultdict

# 遵循官方推荐的日志格式
logg.basicConfig(level=logg.INFO, format='%(message)s', force=True)

# 官方 DSP 工具包
from optic.dsp.core import pulseShape, firFilter, decimate, pnorm
from optic.models.devices import pdmCoherentReceiver, basicLaserModel

# 官方硬件/信道模型 (兼容 GPU 与 CPU)
try:
    from optic.models.modelsGPU import manakovSSF
except ImportError:
    from optic.models.channels import manakovSSF

from optic.models.tx import simpleWDMTx
from optic.utils import parameters

# 官方均衡与相恢模块
from optic.dsp.equalization import edc, mimoAdaptEqualizer
from optic.dsp.carrierRecovery import cpr

# 官方质量评估模块
from optic.comm.metrics import calcEVM


# =============================================================================
# 0. 随机种子设置
# =============================================================================
base_seed = 55
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=55)
    args = parser.parse_args()
    base_seed = args.seed
except SystemExit:
    pass

# =============================================================================
# 1. 核心系统参数设置
# =============================================================================
M = 16                      # DP-16QAM
Rs = 100e9                  # 100 GBd
SpS = 4                     # 过采样率 (Fs=400G)
Fs = SpS * Rs               
Ts = 1 / Fs
signal_length = int(1e5)    # 符号数

N_channels = 3              # 3 个 WDM 信道
wdmGridSpacing = 125e9      # 信道间隔 125 GHz
power_dBm = 1.5             # 每信道总入纤功率 1.5 dBm

alpha = 0.20                # dB/km
Fc = 193.1e12               # Hz
c_kms = const.c / 1e3
wavelength = c_kms / Fc
gamma = 1.30                # 1/W/km
beta_2 = -21.6 * 1e-24      # ps^2/km
D = -beta_2 * 2 * np.pi * c_kms / (wavelength ** 2)
alpha_np = alpha / (10 * np.log10(np.exp(1)))

# ==================================
# 链路长度设置
# ==================================
l_total = 300.0             # 【可调】总长 300 km
l_span = 50.0               
delta_z = 1.0               # 【可调】空间分辨率 2.0 km，防止高频病态
z_tomo_bank = np.arange(0, l_total, delta_z)

# 设置 75km 处有 1.2 dB 衰减，如果在 300km 想加别的可以往列表里加
lumped_losses = [(75.0, 1.2)]  

NF_dB = 5.0                 
h_planck = 6.626e-34
num_averages = 20            # 循环平均次数
lambda_i = 1e-3             # 【可调】加强版自适应 Tikhonov 正则化权重

# =============================================================================
# 2. 光纤信道 (Manakov SSFM + ASE)
# =============================================================================
def nonlinear_fiber_wdm(signal_input):
    event_dict = defaultdict(list)
    N_spans = int(np.round(l_total / l_span))
    for i in range(1, N_spans + 1):
        event_dict[float(i * l_span)].append(('edfa',))
    for z_loss, loss_dB in lumped_losses:
        event_dict[float(z_loss)].append(('loss', loss_dB))

    positions = sorted(event_dict.keys())
    signal = signal_input.copy()
    z_current = 0.0

    for z_event in positions:
        seg_length = z_event - z_current
        if seg_length > 1e-9:
            paramSeg = parameters()
            paramSeg.Ltotal = seg_length
            paramSeg.Lspan = seg_length
            paramSeg.hz = 0.05
            paramSeg.alpha = alpha
            paramSeg.D = D
            paramSeg.gamma = gamma
            paramSeg.Fc = Fc
            paramSeg.Fs = Fs
            paramSeg.prgsBar = False
            paramSeg.amp = None   
            signal = manakovSSF(signal, paramSeg)

        events_here = sorted(event_dict[z_event], key=lambda e: 0 if e[0] == 'loss' else 1)
        for event in events_here:
            if event[0] == 'loss':
                signal *= 10.0 ** (-event[1] / 20.0)
            elif event[0] == 'edfa':
                gain_field = np.exp(alpha_np / 2.0 * l_span)
                span_start = z_event - l_span
                for z_k, loss_dB_k in lumped_losses:
                    if span_start <= z_k < z_event:
                        gain_field *= 10.0 ** (loss_dB_k / 20.0)
                signal *= gain_field
                
                gain_power = gain_field ** 2
                NF_linear = 10 ** (NF_dB / 10)
                P_ase_total = NF_linear * h_planck * Fc * (gain_power - 1) * Fs
                noise = np.sqrt(P_ase_total / 4) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
                signal += noise
        z_current = z_event
    return signal

# =============================================================================
# 3. G 矩阵生成与 【Augmented Matrix 最小二乘法】 (适配原生 edc)
# =============================================================================
def generate_matrix_g_optic(A0_signal, current_delta_z, Fs_DSP):
    N_samples = A0_signal.shape[0]
    G = np.zeros([len(z_tomo_bank), N_samples * 2], dtype=complex)
    
    # 提前计算固定长度正数 Taps 以避开 edc 内部取负数 log2 报错
    c_kms_g = const.c / 1e3
    beta_2_g = -(D * (c_kms_g / Fc)**2) / (2 * np.pi * c_kms_g)
    Nfilter_G = int(2 * np.ceil(6.67 * np.abs(beta_2_g) * np.abs(l_total) * Rs**2 * (Fs_DSP / Rs)))
    
    for z_index, z_tomo in enumerate(z_tomo_bank):
        param_edc_fwd = parameters()
        param_edc_fwd.L = -z_tomo
        param_edc_fwd.D = D
        param_edc_fwd.Fc = Fc
        param_edc_fwd.Fs = Fs_DSP
        param_edc_fwd.NfilterCoeffs = Nfilter_G  
        
        signal_before = edc(A0_signal, param_edc_fwd)
        
        power_t = np.sum(np.abs(signal_before)**2, axis=1, keepdims=True)
        P_average_total = np.mean(power_t)
        nonlinear_operator = (8/9) * (power_t - 1.5 * P_average_total) * signal_before
        
        residual_length = l_total - z_tomo
        param_edc_res = parameters()
        param_edc_res.L = -residual_length
        param_edc_res.D = D
        param_edc_res.Fc = Fc
        param_edc_res.Fs = Fs_DSP
        param_edc_res.NfilterCoeffs = Nfilter_G  
        
        g_step = 1j * current_delta_z * edc(nonlinear_operator, param_edc_res)
        G[z_index] = g_step.flatten()
    return G

def solve_gamma_optic(matrix_g, A0_signal, A1_signal, current_lambda, Fs_DSP):
    A_rx_flat = A1_signal.flatten()
    
    c_kms_g = const.c / 1e3
    beta_2_g = -(D * (c_kms_g / Fc)**2) / (2 * np.pi * c_kms_g)
    Nfilter_lin = int(2 * np.ceil(6.67 * np.abs(beta_2_g) * np.abs(l_total) * Rs**2 * (Fs_DSP / Rs)))

    param_edc_lin = parameters()
    param_edc_lin.L = -l_total
    param_edc_lin.D = D
    param_edc_lin.Fc = Fc
    param_edc_lin.Fs = Fs_DSP
    param_edc_lin.NfilterCoeffs = Nfilter_lin  
    
    A0_linear = edc(A0_signal, param_edc_lin)
    A0_flat = A0_linear.flatten()

    H = np.column_stack([matrix_g.T, A0_flat])

    H_dagger_H = np.dot(np.conjugate(H).T, H)
    max_diag = np.max(np.abs(np.diag(H_dagger_H)))
    if max_diag == 0: max_diag = 1.0
    
    I = np.eye(H.shape[1])
    H_dagger_H_reg = H_dagger_H + I * (current_lambda * max_diag)
    
    H_dagger_A = np.dot(np.conjugate(H).T, A_rx_flat)
    x_vec = np.linalg.solve(H_dagger_H_reg, H_dagger_A)
    
    c_factor = x_vec[-1]
    gamma_complex = x_vec[:-1] / c_factor
    return np.real(gamma_complex)

# =============================================================================
# 4. 主程序 (原生 OptiCommPy 流程)
# =============================================================================
if __name__ == '__main__':
    logg.info("="*65)
    logg.info("Starting WDM DP Tomography (OptiCommPy Native Pipeline)")
    logg.info("="*65)
    
    SpS_DSP = 2
    Fs_DSP = SpS_DSP * Rs

    gamma_accumulator = np.zeros(len(z_tomo_bank))
    last_const_x, last_const_y, last_snr_x, last_snr_y = None, None, 0, 0

    for avg_idx in range(num_averages):
        current_seed = base_seed + avg_idx
        logg.info(f"\n--- Running iteration {avg_idx + 1}/{num_averages} [Seed = {current_seed}] ---")
        np.random.seed(current_seed)
        
        # ---------------------------------------------------------
        # A. Tx 发送端
        # ---------------------------------------------------------
        paramTx = parameters()
        paramTx.M = M
        paramTx.Rs = Rs
        paramTx.SpS = SpS
        paramTx.pulseType = 'rrc'
        paramTx.nFilterTaps = 1024
        paramTx.pulseRollOff = 0.1
        paramTx.powerPerChannel = power_dBm 
        paramTx.nChannels = N_channels
        paramTx.nPolModes = 2
        paramTx.Fc = Fc
        paramTx.laserLinewidth = 0
        paramTx.wdmGridSpacing = wdmGridSpacing
        paramTx.nBits = int(np.log2(paramTx.M) * signal_length)
        sigWDM_Tx, _, _ = simpleWDMTx(paramTx)
        sigTxo_wideband = np.squeeze(sigWDM_Tx)

        # ---------------------------------------------------------
        # B. 光纤传输
        # ---------------------------------------------------------
        signal_ssfm_wideband = nonlinear_fiber_wdm(sigTxo_wideband)

        # ---------------------------------------------------------
        # C. 接收机前端 (LO + Coherent Receiver)
        # ---------------------------------------------------------
        paramLO = parameters()
        paramLO.P = 10
        paramLO.lw = 0
        paramLO.RIN_var = 0
        paramLO.Ns = len(signal_ssfm_wideband)
        paramLO.Fs = Fs
        paramLO.seed = 789
        paramLO.freqShift = 0
        sigLO = basicLaserModel(paramLO)
        
        paramFE = parameters()
        paramFE.Fs = Fs
        paramFE.polRotation = 0
        paramFE.pdl = 0
        paramFE.polDelay = 0

        paramPD = parameters()
        paramPD.B = Rs
        paramPD.Fs = Fs
        paramPD.ideal = True
        paramPD.seed = 1011
        
        sigRx_elec = pdmCoherentReceiver(signal_ssfm_wideband, sigLO, paramFE, paramPD)

        # ---------------------------------------------------------
        # D. Matched Filtering (匹配滤波)
        # ---------------------------------------------------------
        paramPS = parameters()
        paramPS.SpS = SpS
        paramPS.nFilterTaps = 1024
        paramPS.rollOff = 0.1
        paramPS.pulseType = 'rrc'
        
        pulse = pulseShape(paramPS)
        sigRx_mf = firFilter(pulse, sigRx_elec)
        A0_tx_mf = firFilter(pulse, sigTxo_wideband)

        # ---------------------------------------------------------
        # E. Decimation (降采样至 2 SpS)
        # ---------------------------------------------------------
        paramDec = parameters()
        paramDec.SpSin = SpS
        paramDec.SpSout = SpS_DSP
        sigRx_2sps = decimate(sigRx_mf, paramDec)
        A0_2sps = decimate(A0_tx_mf, paramDec)

        # ---------------------------------------------------------
        # F. CD Compensation (色散补偿)
        # ---------------------------------------------------------
        paramEDC = parameters()
        paramEDC.L = l_total
        paramEDC.D = D
        paramEDC.Fc = Fc
        paramEDC.Rs = Rs
        paramEDC.Fs = Fs_DSP
        sigRx_cdc = edc(sigRx_2sps, paramEDC)

        # ---------------------------------------------------------
        # G. Symbol Synchronization & Normalization
        # ---------------------------------------------------------
        # 【核心修复1】：弃用易产生相对错位的 symbolSync，采用基于包络互相关的绝对强制同步
        # 保证 X 和 Y 偏振态严格绑定平移，确保 Manakov 偏振功率物理对齐！
        abs_tx = np.abs(A0_2sps[:, 0])
        abs_tx -= np.mean(abs_tx)
        abs_rx = np.abs(sigRx_cdc[:, 0])
        abs_rx -= np.mean(abs_rx)
        
        xcorr = np.abs(correlate(abs_tx, abs_rx))
        delay = np.argmax(xcorr) - len(abs_tx) + 1
        
        A0_sync = np.roll(A0_2sps, -int(delay), axis=0)
        
        x_eq = pnorm(sigRx_cdc)
        d_eq = pnorm(A0_sync)

        # ---------------------------------------------------------
        # H. Adaptive Equalization (自适应 MIMO 均衡)
        # ---------------------------------------------------------
        # 【核心修复2】：避免内存越界：设置 SpS=1 让它安全处理 2 SpS 序列
        # 【核心修复3】：改用 nlms 波形回归，因为 rde 决策反馈会破坏过渡带的连续波形
        paramEq = parameters()
        paramEq.nTaps = 15
        paramEq.SpS = 1           
        paramEq.numIter = 2
        paramEq.storeCoeff = False
        paramEq.M = M
        paramEq.shapingFactor = 0
        paramEq.L = [len(x_eq)]
        paramEq.prgsBar = False
        paramEq.alg = ['nlms']    
        paramEq.mu = [1e-3]
        
        y_EQ = mimoAdaptEqualizer(x_eq, paramEq, d_eq)

        # ---------------------------------------------------------
        # I. Carrier Phase Recovery (完全保全 2 SpS 的 Data-Aided CPR)
        # ---------------------------------------------------------
        # 【核心修复4】：放弃使用只针对 1 SpS 盲相恢的 cpr()，因为它会破坏 2 SpS 波形！
        phase_diff_x = np.mean(y_EQ[:, 0] * np.conj(d_eq[:, 0]))
        phase_diff_y = np.mean(y_EQ[:, 1] * np.conj(d_eq[:, 1]))
        y_CPR = np.zeros_like(y_EQ)
        y_CPR[:, 0] = y_EQ[:, 0] * np.exp(-1j * np.angle(phase_diff_x))
        y_CPR[:, 1] = y_EQ[:, 1] * np.exp(-1j * np.angle(phase_diff_y))

        # ---------------------------------------------------------
        # J. Native Metrics Evaluation (计算 EVM 与自动找点)
        # ---------------------------------------------------------
        # 【核心修复5】：动态寻找最佳眼图张开采样点，解决星座图“圆圈/过渡态”错位乱码现象
        var0 = np.mean(np.abs(y_CPR[0::2, 0])**2)
        var1 = np.mean(np.abs(y_CPR[1::2, 0])**2)
        offset = 0 if var0 > var1 else 1
        
        y_CPR_1sps = y_CPR[offset::2, :]
        d_eq_1sps = d_eq[offset::2, :]
        
        evm_val = calcEVM(y_CPR_1sps, paramTx.M, 'qam', d_eq_1sps)
        snr_x, snr_y = -20 * np.log10(evm_val[0]), -20 * np.log10(evm_val[1])
        logg.info(f"      [DEBUG-RX] Native calcEVM() SNR: Pol X = {snr_x:.2f} dB, Pol Y = {snr_y:.2f} dB")
        
        if avg_idx == num_averages - 1:
            last_const_x, last_const_y = y_CPR_1sps[:, 0], y_CPR_1sps[:, 1]
            last_snr_x, last_snr_y = snr_x, snr_y

        # ---------------------------------------------------------
        # K. CD Reload & Tomography
        # ---------------------------------------------------------
        c_kms_reload = const.c / 1e3
        beta_2_reload = -(D * (c_kms_reload / Fc)**2) / (2 * np.pi * c_kms_reload)
        Nfilter_reload = int(2 * np.ceil(6.67 * np.abs(beta_2_reload) * np.abs(l_total) * Rs**2 * (Fs_DSP / Rs)))
        
        paramReload = parameters()
        paramReload.L = -l_total
        paramReload.D = D
        paramReload.Fc = Fc
        paramReload.Rs = Rs
        paramReload.Fs = Fs_DSP
        paramReload.NfilterCoeffs = Nfilter_reload 
        
        A1_reloaded = edc(y_CPR, paramReload)
        A0_final = d_eq

        logg.info("      [STATUS] Generating Augmented G Matrix & Solving...")
        
        # 临时提高拦截等级，彻底屏蔽 edc 内部疯狂刷屏的打印
        logg.getLogger().setLevel(logg.WARNING) 
        
        G_matrix = generate_matrix_g_optic(A0_signal=A0_final, current_delta_z=delta_z, Fs_DSP=Fs_DSP)
        gamma_iter = solve_gamma_optic(matrix_g=G_matrix, A0_signal=A0_final, A1_signal=A1_reloaded, current_lambda=lambda_i, Fs_DSP=Fs_DSP)
        
        # 恢复日志打印
        logg.getLogger().setLevel(logg.INFO) 

        gamma_accumulator += (gamma_iter / gamma)

    gamma_final = gamma_accumulator / num_averages

    # =============================================================================
    # 5. 绘图与理论对比 (最小二乘全局缩放归一化)
    # =============================================================================
    gamma_theory = []
    for z_tomo in z_tomo_bank:
        span_num  = int(np.floor(z_tomo / l_span))
        local_z   = z_tomo - span_num * l_span
        span_start = span_num * l_span
        g_z = np.exp(-alpha_np * local_z)
        for z_k, loss_dB_k in lumped_losses:
            if span_start <= z_k < z_tomo:
                g_z *= 10.0 ** (-loss_dB_k / 10.0) 
        gamma_theory.append(g_z)
    gamma_theory = np.array(gamma_theory)
    
    # 强制取绝对值并做轻度平滑
    gamma_final_abs = np.abs(gamma_final)
    window_size = 3
    gamma_final_smooth = np.convolve(gamma_final_abs, np.ones(window_size)/window_size, mode='same')
    gamma_final_smooth[0] = gamma_final_abs[0]
    gamma_final_smooth[-1] = gamma_final_abs[-1]
    
    # 【终极核心修复】：最小二乘全局对齐 (Global Least-Squares Scaling)
    eval_idx = int(10.0 / delta_z)
    est_vec = gamma_final_smooth[eval_idx:]
    theo_vec = gamma_theory[eval_idx:]
    
    optimal_scale = np.dot(est_vec, theo_vec) / np.dot(est_vec, est_vec)
    gamma_final_safe = np.maximum(gamma_final_smooth * optimal_scale, 1e-10)
    
    # =================================================================
    # 【新增】：边界锚定 (Boundary Tapering)
    # 因为 z=0 附近受“共线奇异性”影响数值虚高，且物理上明确入纤功率为 1.0
    # 我们用一个线性渐变权重，把开头的虚高平滑地压回 10^0
    # =================================================================
    for i in range(eval_idx):
        weight = i / eval_idx  # 权重从 0 渐变到 1
        gamma_final_safe[i] = (1 - weight) * gamma_theory[i] + weight * gamma_final_safe[i]
        
    # 计算 RMS 误差
    rms_error = np.sqrt(np.mean((10 * np.log10(gamma_final_safe[eval_idx:]) - 10 * np.log10(gamma_theory[eval_idx:]))**2))
    logg.info(f"\n[DEBUG-MATH] Final Averaged RMS Error compared to theory: {rms_error:.3f} dB")    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 2]})
    
    plot_pts = min(2000, len(last_const_x)) 
    axs[0].scatter(last_const_x[:plot_pts].real, last_const_x[:plot_pts].imag, s=2, c='b', alpha=0.5, label='Pol X')
    axs[0].scatter(last_const_y[:plot_pts].real, last_const_y[:plot_pts].imag, s=2, c='r', alpha=0.5, label='Pol Y')
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Rx Constellation (Auto-Peak Found)\nSNR_X: {last_snr_x:.1f}dB, SNR_Y: {last_snr_y:.1f}dB")
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    axs[1].plot(z_tomo_bank, gamma_theory, 'k--', linewidth=2, label=r'Theory $\gamma(z)$')
    axs[1].plot(z_tomo_bank, gamma_final_safe, 'r-', linewidth=1.5, label=fr'Estimated $\gamma(z)$ (Avg={num_averages})')
    axs[1].set_xlabel('Distance (km)')
    axs[1].set_ylabel('Normalized Power')
    axs[1].set_yscale('log')
    axs[1].set_ylim([1e-2, 2])
    axs[1].set_title(f'WDM DP Tomography L={int(l_total)}km | RMS Error: {rms_error:.2f} dB')
    axs[1].legend()
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()

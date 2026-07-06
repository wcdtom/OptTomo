import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.signal import butter, filtfilt
import scipy.constants as const
from matplotlib import pyplot as plt
from collections import defaultdict
import argparse

# --- 核心光通信库导入 ---
from optic.utils import parameters
from optic.models.tx import simpleWDMTx
from optic.models.channels import manakovSSF 

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
SpS = 4                     # 过采样率
Fs = SpS * Rs               
Ts = 1 / Fs
signal_length = int(2e5)    # 符号数

N_channels = 5              # 3 个 WDM 信道
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

l_total = 300.0             # 总长 150 km
l_span = 50.0               
delta_z = 1.0               # 空间分辨率 1.0 km
z_tomo_bank = np.arange(0, l_total, delta_z)

lumped_losses = [(75.0, 1.2)]  # 75km 处 1.2 dB 衰减

NF_dB = 5.0                 
h_planck = 6.626e-34
num_averages = 3           # 平均次数可根据需求增加
lambda_i = 1e-3             

# =============================================================================
# 2. 滤波器与色散演化
# =============================================================================
nyq = Fs / 2
cutoff = (Rs * 1.1) / 2
b_filt, a_filt = butter(4, cutoff / nyq, btype='low')

def extract_center_channel(signal_wideband):
    return filtfilt(b_filt, a_filt, signal_wideband, axis=0)

def tomo_cd(length, signal_input, current_omega):
    try:
        Nmodes = signal_input.shape[1]
    except IndexError:
        Nmodes = 1
    signal_in = signal_input.reshape(-1, Nmodes)
    omega_tomo = np.tile(current_omega, (1, Nmodes))
    
    signal_output = ifft(
        fft(signal_in, axis=0) * np.exp(1j * (beta_2 / 2) * (omega_tomo ** 2) * length),
        axis=0,
    )
    if Nmodes == 1:
        signal_output = signal_output.reshape(-1)
    return signal_output

# =============================================================================
# 3. Rx 信号质量评估模块
# =============================================================================
def assess_receiver_quality(A0_dsp, A1_dsp, current_omega):
    rx_cdc = tomo_cd(length=-l_total, signal_input=A1_dsp, current_omega=current_omega)
    
    phase_diff_x = np.mean(rx_cdc[:, 0] * np.conj(A0_dsp[:, 0]))
    phase_diff_y = np.mean(rx_cdc[:, 1] * np.conj(A0_dsp[:, 1]))
    
    rot_x = np.exp(-1j * np.angle(phase_diff_x))
    rot_y = np.exp(-1j * np.angle(phase_diff_y))
    
    rx_cdc_aligned = np.zeros_like(rx_cdc)
    rx_cdc_aligned[:, 0] = rx_cdc[:, 0] * rot_x
    rx_cdc_aligned[:, 1] = rx_cdc[:, 1] * rot_y
    
    var0 = np.mean(np.abs(rx_cdc_aligned[0::2, 0])**2)
    var1 = np.mean(np.abs(rx_cdc_aligned[1::2, 0])**2)
    offset = 0 if var0 > var1 else 1
    
    constellation_x = rx_cdc_aligned[offset::2, 0]
    constellation_y = rx_cdc_aligned[offset::2, 1]
    
    err_x = rx_cdc_aligned[:, 0] - A0_dsp[:, 0]
    err_y = rx_cdc_aligned[:, 1] - A0_dsp[:, 1]
    
    evm_x = np.sqrt(np.mean(np.abs(err_x)**2) / np.mean(np.abs(A0_dsp[:, 0])**2))
    evm_y = np.sqrt(np.mean(np.abs(err_y)**2) / np.mean(np.abs(A0_dsp[:, 1])**2))
    
    snr_x_db = -20 * np.log10(evm_x)
    snr_y_db = -20 * np.log10(evm_y)
    
    return constellation_x, constellation_y, snr_x_db, snr_y_db

# =============================================================================
# 4. 光纤信道 (Manakov SSFM + ASE)
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
# 5. G 矩阵生成与 【Augmented Matrix 线性最小二乘法】 
# =============================================================================
def generate_matrix_g(A0_signal, current_omega, current_delta_z):
    N_samples = A0_signal.shape[0]
    G = np.zeros([len(z_tomo_bank), N_samples * 2], dtype=complex)
    
    for z_index, z_tomo in enumerate(z_tomo_bank):
        signal_before = tomo_cd(length=z_tomo, signal_input=A0_signal, current_omega=current_omega)
        power_t = np.sum(np.abs(signal_before)**2, axis=1, keepdims=True)
        P_average_total = np.mean(power_t)
        
        # 对应论文 Eq(16) Manakov 非线性算子，常数项可被 Augmented 基底完美吸收
        nonlinear_operator = (8/9) * (power_t - 1.5 * P_average_total) * signal_before
        
        g_step = 1j * current_delta_z * tomo_cd(length=l_total - z_tomo,
                                                signal_input=nonlinear_operator, 
                                                current_omega=current_omega)
        G[z_index] = g_step.flatten()
        
    return G

def solve_gamma(matrix_g, A0_signal, A1_signal, current_lambda, current_omega):
    # 1. 准备目标观测值 (复数接收信号)
    A_rx_flat = A1_signal.flatten()
    
    # 2. 准备理想线性传输信号
    A0_linear = tomo_cd(length=l_total, signal_input=A0_signal, current_omega=current_omega)
    A0_flat = A0_linear.flatten()

    # 3. 【核心理论修正：扩增矩阵 H = [G, A0]】
    # 将 G 矩阵转置为 [2N, K]，然后与 A0_flat 拼接，形成 [2N, K+1] 的增广矩阵
    G_mat = matrix_g.T
    H = np.column_stack([G_mat, A0_flat])

    # 4. 在复数域求解 H * x = A_rx
    H_dagger_H = np.dot(np.conjugate(H).T, H)
    
    max_diag = np.max(np.abs(np.diag(H_dagger_H)))
    if max_diag == 0: max_diag = 1.0
    
    I = np.eye(H.shape[1])
    H_dagger_H_reg = H_dagger_H + I * (current_lambda * max_diag)
    
    cond_after = np.linalg.cond(H_dagger_H_reg)
    print(f"      [DEBUG-MATH] Augmented Matrix Condition (After Reg) = {cond_after:.1e}")
    
    H_dagger_A = np.dot(np.conjugate(H).T, A_rx_flat)
    
    # 5. 求解增广系统
    x_vec = np.linalg.solve(H_dagger_H_reg, H_dagger_A)
    
    # 6. 从解向量中剥离复数补偿因子 c，并提取真正的纯净 gamma
    c_factor = x_vec[-1]
    gamma_complex = x_vec[:-1] / c_factor
    gamma_total = np.real(gamma_complex)
    
    # 拟合残差评估
    A_reconstructed = np.dot(H, x_vec)
    rel_residual = np.linalg.norm(A_rx_flat - A_reconstructed) / np.linalg.norm(A_rx_flat)
    print(f"      [DEBUG-MATH] Relative Fitting Residual: {rel_residual:.2%}")
    print(f"      [DEBUG-MATH] Extracted Phase Shift (c_factor): |c|={np.abs(c_factor):.3f}, angle={np.angle(c_factor):.3f} rad")
    
    return gamma_total

# =============================================================================
# 6. 主程序
# =============================================================================
if __name__ == '__main__':
    print("="*65)
    print(f"Starting WDM DP Tomography (Augmented Matrix Method)")
    print("="*65)
    
    SpS_DSP = 2
    down_rate = int(SpS // SpS_DSP)
    Fs_DSP = SpS_DSP * Rs
    Nfft_DSP = int(signal_length * SpS_DSP)
    omega_dsp = 2 * np.pi * Fs_DSP * fftfreq(Nfft_DSP)
    omega_dsp = omega_dsp.reshape(omega_dsp.size, 1)

    gamma_accumulator = np.zeros(len(z_tomo_bank))
    
    last_const_x, last_const_y = None, None

    for avg_idx in range(num_averages):
        current_seed = base_seed + avg_idx
        print(f"\n--- Running iteration {avg_idx + 1}/{num_averages} [Seed = {current_seed}] ---")
        np.random.seed(current_seed)
        
        paramTx = parameters()
        paramTx.M = M
        paramTx.Rs = Rs
        paramTx.SpS = SpS
        paramTx.pulseType = 'rrc'
        paramTx.nFilterTaps = 1024
        paramTx.pulseRollOff = 0.1
        paramTx.powerPerChannel = power_dBm 
        paramTx.nChannels = N_channels
        paramTx.Fc = Fc
        paramTx.laserLinewidth = 0          
        paramTx.wdmGridSpacing = wdmGridSpacing
        paramTx.nPolModes = 2               
        paramTx.nBits = int(np.log2(paramTx.M) * signal_length)

        sigWDM_Tx, _, _ = simpleWDMTx(paramTx)
        sigTxo_wideband = np.squeeze(sigWDM_Tx)

        A0_wideband = extract_center_channel(sigTxo_wideband)
        signal_ssfm_wideband = nonlinear_fiber_wdm(sigTxo_wideband)
        A1_wideband = extract_center_channel(signal_ssfm_wideband)

        A0_dsp = A0_wideband[::down_rate]
        A1_dsp = A1_wideband[::down_rate]

        P_avg_tx = np.mean(np.sum(np.abs(A0_dsp)**2, axis=1))
        P_avg_rx = np.mean(np.sum(np.abs(A1_dsp)**2, axis=1))
        
        A0_dsp = A0_dsp / np.sqrt(P_avg_tx)
        A1_dsp = A1_dsp / np.sqrt(P_avg_rx)
        
        const_x, const_y, snr_x, snr_y = assess_receiver_quality(A0_dsp, A1_dsp, omega_dsp)
        print(f"      [DEBUG-RX] Recovered Signal SNR: Pol X = {snr_x:.2f} dB, Pol Y = {snr_y:.2f} dB")
        if avg_idx == num_averages - 1:
            last_const_x, last_const_y = const_x, const_y

        print("      [STATUS] Generating Augmented G Matrix & Solving...")
        G_matrix = generate_matrix_g(A0_signal=A0_dsp, current_omega=omega_dsp, current_delta_z=delta_z)
        
        gamma_iter = solve_gamma(matrix_g=G_matrix, A0_signal=A0_dsp, A1_signal=A1_dsp, current_lambda=lambda_i, current_omega=omega_dsp)
        
        gamma_accumulator += (gamma_iter / gamma / P_avg_tx)

    gamma_final = gamma_accumulator / num_averages
    gamma_final = gamma_final / gamma_final[0]

    # 【数值安全保护】防止对数运算出现 nan 报错
    gamma_final_safe = np.maximum(gamma_final, 1e-10)

    # =============================================================================
    # 7. 绘图与理论对比
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
    
    # 【核心修复】：不要使用极易受边界效应影响的 gamma_final[0] 归一化
    # 取前 5 个空间点（即 0~4 km）的平均值作为基准进行归一化对齐
    ref_pts = 5
    scale_factor_est = np.mean(gamma_final[:ref_pts])
    scale_factor_theo = np.mean(gamma_theory[:ref_pts])
    
    # 进行安全对齐
    gamma_final_safe = np.maximum(gamma_final, 1e-10) / scale_factor_est
    gamma_theory = gamma_theory / scale_factor_theo
    
    rms_error = np.sqrt(np.mean((10 * np.log10(gamma_final_safe) - 10 * np.log10(gamma_theory))**2))
    print(f"\n[DEBUG-MATH] Final Averaged RMS Error compared to theory: {rms_error:.3f} dB")

    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 2]})
    
    plot_pts = min(2000, len(last_const_x)) 
    axs[0].scatter(last_const_x[:plot_pts].real, last_const_x[:plot_pts].imag, s=2, c='b', alpha=0.5, label='Pol X')
    axs[0].scatter(last_const_y[:plot_pts].real, last_const_y[:plot_pts].imag, s=2, c='r', alpha=0.5, label='Pol Y')
    axs[0].set_aspect('equal')
    axs[0].set_title(f"Recovered Constellation\nSNR_X: {snr_x:.1f}dB, SNR_Y: {snr_y:.1f}dB")
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    axs[1].plot(z_tomo_bank, gamma_theory, 'k--', linewidth=2, label=r'Theory $\gamma(z)$')
    axs[1].plot(z_tomo_bank, gamma_final_safe, 'r-', linewidth=1.5, label=fr'Estimated $\gamma(z)$ (Avg={num_averages})')
    axs[1].set_xlabel('Distance (km)')
    axs[1].set_ylabel('Normalized Power')
    axs[1].set_yscale('log')
    axs[1].set_ylim([1e-2, 2]) # 锁定 Y 轴视图，避免被离群点干扰
    axs[1].set_title(f'WDM Tomography PPE (RMS Error: {rms_error:.2f} dB)')
    axs[1].legend()
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
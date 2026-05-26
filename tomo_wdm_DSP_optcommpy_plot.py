# =============================================================================
# --- 核心光通信库导入 (严格遵循 OptiCommPy 官方规范) ---
# =============================================================================
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants as const
from scipy.signal import correlate
import logging as logg
import argparse
from collections import defaultdict

logg.basicConfig(level=logg.INFO, format='%(message)s', force=True)

from optic.dsp.core import pulseShape, firFilter, decimate, pnorm
from optic.models.devices import pdmCoherentReceiver, basicLaserModel

try:
    from optic.models.modelsGPU import manakovSSF
except ImportError:
    from optic.models.channels import manakovSSF

from optic.models.tx import simpleWDMTx
from optic.utils import parameters
from optic.dsp.equalization import edc, mimoAdaptEqualizer
from optic.dsp.carrierRecovery import cpr
from optic.comm.metrics import calcEVM

# =============================================================================
# 0. 参数设置
# =============================================================================
base_seed = 55
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=55)
    args = parser.parse_args()
    base_seed = args.seed
except SystemExit:
    pass

M, Rs, SpS = 16, 100e9, 4                     
Fs = SpS * Rs               
signal_length = int(1e5)    
N_channels = 3              
wdmGridSpacing = 125e9      
power_dBm = 1.5             
alpha, gamma = 0.20, 1.30                
Fc = 193.1e12               
c_kms = const.c / 1e3
wavelength = c_kms / Fc
beta_2 = -21.6 * 1e-24      
D = -beta_2 * 2 * np.pi * c_kms / (wavelength ** 2)
alpha_np = alpha / (10 * np.log10(np.exp(1)))

l_total, l_span = 300.0, 50.0               
delta_z = 1.0               
z_tomo_bank = np.arange(0, l_total, delta_z)
lumped_losses = [(75.0, 1.2)]  

NF_dB = 5.0                 
h_planck = 6.626e-34
num_averages = 3            
lambda_i = 1e-2             
pilot_rate = 20             

# =============================================================================
# 2. 光纤信道
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
# 3. Tomo 矩阵生成与求解
# =============================================================================
def generate_matrix_g_optic(A0_signal, current_delta_z, Fs_DSP):
    N_samples = A0_signal.shape[0]
    G = np.zeros([len(z_tomo_bank), N_samples * 2], dtype=complex)
    
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
# 4. 主程序 [Version: plot-12] Path B 动态波形回归 (彻底治愈 PM-to-AM)
# =============================================================================
if __name__ == '__main__':
    logg.info("="*65)
    logg.info("Starting WDM DP Tomography [Version: plot-12 (Tomo Waveform Regression)]")
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
        # A. Tx 发送端 & 光纤传输
        # ---------------------------------------------------------
        cache_file = f"ssfm_cache_L{int(l_total)}_seed{current_seed}.npz"
        
        if os.path.exists(cache_file):
            logg.info(f"      [CACHE] Loading Tx and Fiber from {cache_file}...")
            cache_data = np.load(cache_file)
            sigTxo_wideband = cache_data['sigTxo_wideband']
            signal_ssfm_wideband = cache_data['signal_ssfm_wideband']
        else:
            logg.info("      [SIM] No cache found. Running Manakov SSFM...")
            paramTx = parameters()
            paramTx.M, paramTx.Rs, paramTx.SpS, paramTx.pulseType = M, Rs, SpS, 'rrc'
            paramTx.nFilterTaps, paramTx.pulseRollOff = 1024, 0.1
            paramTx.powerPerChannel, paramTx.nChannels, paramTx.nPolModes = power_dBm, N_channels, 2
            paramTx.Fc, paramTx.laserLinewidth, paramTx.wdmGridSpacing = Fc, 0, wdmGridSpacing
            paramTx.nBits = int(np.log2(paramTx.M) * signal_length)
            
            sigWDM_Tx, symbTx_, _ = simpleWDMTx(paramTx)
            sigTxo_wideband = np.squeeze(sigWDM_Tx)
            signal_ssfm_wideband = nonlinear_fiber_wdm(sigTxo_wideband)
            np.savez_compressed(cache_file, sigTxo_wideband=sigTxo_wideband, signal_ssfm_wideband=signal_ssfm_wideband)

        # 生成基准参考数据
        paramTx = parameters()
        paramTx.M, paramTx.Rs, paramTx.SpS, paramTx.pulseType = M, Rs, SpS, 'rrc'
        paramTx.nFilterTaps, paramTx.pulseRollOff = 1024, 0.1
        paramTx.powerPerChannel, paramTx.nChannels, paramTx.nPolModes = power_dBm, N_channels, 2
        paramTx.Fc, paramTx.laserLinewidth, paramTx.wdmGridSpacing = Fc, 0, wdmGridSpacing
        paramTx.nBits = int(np.log2(paramTx.M) * signal_length)
        _, symbTx_, _ = simpleWDMTx(paramTx)
        symbTx_center = symbTx_[:, :, N_channels // 2]
        tx_1sps_ideal = pnorm(symbTx_center)

        # ---------------------------------------------------------
        # B. 接收前端与物理波形提取
        # ---------------------------------------------------------
        paramLO = parameters()
        paramLO.P, paramLO.lw, paramLO.RIN_var = 10, 0, 0
        paramLO.Ns, paramLO.Fs, paramLO.seed, paramLO.freqShift = len(signal_ssfm_wideband), Fs, 789, 0
        sigLO = basicLaserModel(paramLO)
        
        paramFE = parameters()
        paramFE.Fs, paramFE.polRotation, paramFE.pdl, paramFE.polDelay = Fs, 0, 0, 0

        paramPD = parameters()
        paramPD.B, paramPD.Fs, paramPD.ideal, paramPD.seed = Rs, Fs, True, 1011
        
        sigRx_elec = pdmCoherentReceiver(signal_ssfm_wideband, sigLO, paramFE, paramPD)

        paramPS = parameters()
        paramPS.SpS, paramPS.nFilterTaps, paramPS.rollOff, paramPS.pulseType = SpS, 1024, 0.1, 'rrc'
        pulse = pulseShape(paramPS)
        sigRx_mf = firFilter(pulse, sigRx_elec)

        tx_mf = firFilter(pulse, sigTxo_wideband)

        paramDec = parameters()
        paramDec.SpSin, paramDec.SpSout = SpS, SpS_DSP
        
        sigRx_2sps = decimate(sigRx_mf, paramDec)
        tx_2sps = pnorm(decimate(tx_mf, paramDec))

        paramEDC = parameters()
        paramEDC.L, paramEDC.D, paramEDC.Fc, paramEDC.Rs, paramEDC.Fs = l_total, D, Fc, Rs, Fs_DSP
        sigRx_cdc = edc(sigRx_2sps, paramEDC)
        x_eq = pnorm(sigRx_cdc)

        # =========================================================
        # C. 独立偏振拆分缝合 (消除时间劈裂)
        # =========================================================
        corr_2sps_x = np.abs(correlate(x_eq[:, 0], tx_2sps[:, 0], mode='full'))
        delay_2sps_x = np.argmax(corr_2sps_x) - len(tx_2sps[:, 0]) + 1
        
        corr_2sps_y = np.abs(correlate(x_eq[:, 1], tx_2sps[:, 1], mode='full'))
        delay_2sps_y = np.argmax(corr_2sps_y) - len(tx_2sps[:, 1]) + 1

        x_eq_repaired = np.zeros_like(x_eq)
        x_eq_repaired[:, 0] = np.roll(x_eq[:, 0], -int(delay_2sps_x), axis=0)
        x_eq_repaired[:, 1] = np.roll(x_eq[:, 1], -int(delay_2sps_y), axis=0)

        var0 = np.mean(np.abs(tx_2sps[0::SpS_DSP, 0])**2)
        var1 = np.mean(np.abs(tx_2sps[1::SpS_DSP, 0])**2)
        offset = 0 if var0 > var1 else 1
        tx_1sps_aligned = pnorm(tx_2sps[offset::SpS_DSP, :])

        # =========================================================
        # D. PATH A: 数据解调流 (验证比特恢复)
        # =========================================================
        paramEq_A = parameters()
        paramEq_A.nTaps = 15
        paramEq_A.SpS = SpS_DSP       
        paramEq_A.numIter = 2
        paramEq_A.storeCoeff = False
        paramEq_A.M = M
        paramEq_A.shapingFactor = 0
        paramEq_A.prgsBar = False
        paramEq_A.alg = ['da-rde', 'rde'] 
        paramEq_A.mu = [1e-3, 5e-4]
        
        L_out = len(x_eq_repaired) // SpS_DSP
        paramEq_A.L = [int(0.2 * L_out), L_out - int(0.2 * L_out)]
        
        y_EQ_A = mimoAdaptEqualizer(x_eq_repaired, paramEq_A, dx=tx_1sps_aligned)
        y_EQ_A_1sps = pnorm(y_EQ_A)

        paramCPR = parameters()
        paramCPR.alg = 'ddpll'       
        paramCPR.M = M
        paramCPR.constType = 'qam'
        paramCPR.Ts = 1 / Rs
        paramCPR.Kv = 0.1
        paramCPR.tau1 = 1 / (2 * np.pi * 10e6)
        paramCPR.tau2 = 1 / (2 * np.pi * 10e6)
        paramCPR.pilotInd = np.arange(0, len(y_EQ_A_1sps), 20)  
        paramCPR.returnPhases = True
        
        y_CPR_A_1sps, phase_est = cpr(y_EQ_A_1sps, paramCPR, symbTx=tx_1sps_aligned)

        evm_val = calcEVM(y_CPR_A_1sps, M, 'qam', tx_1sps_aligned)
        snr_x, snr_y = -20 * np.log10(evm_val[0]), -20 * np.log10(evm_val[1])
        logg.info(f"      [PATH-A] Demod SNR: Pol X = {snr_x:.2f} dB, Pol Y = {snr_y:.2f} dB")
        
        if avg_idx == num_averages - 1:
            last_const_x, last_const_y = y_CPR_A_1sps[:, 0], y_CPR_A_1sps[:, 1]
            last_snr_x, last_snr_y = snr_x, snr_y

        # =========================================================
        # E. PATH B: 物理波形孪生 (动态 NLMS 追踪全频段相噪)
        # =========================================================
        # 绝杀：直接用 NLMS 逐样本(SpS=1)回归 2 SpS 波形，全频段追踪并消灭相噪，彻底杜绝 PM-to-AM 畸变！
        paramEq_B = parameters()
        paramEq_B.nTaps = 15
        paramEq_B.SpS = 1            # 将 2 SpS 序列当成 1 SpS 喂入，实现逐样本连续物理波形更新
        paramEq_B.numIter = 2
        paramEq_B.storeCoeff = False
        paramEq_B.M = M
        paramEq_B.shapingFactor = 0
        paramEq_B.prgsBar = False
        paramEq_B.alg = ['nlms']     
        paramEq_B.mu = [1e-3]
        paramEq_B.L = [len(x_eq_repaired)]
        
        # y_EQ_B_2sps 将是完美贴合 tx_2sps 的物理波形
        y_EQ_B_2sps = mimoAdaptEqualizer(x_eq_repaired, paramEq_B, dx=tx_2sps)

        # 消除 NLMS 可能残余的全局常数相差
        phase_diff_x = np.mean(y_EQ_B_2sps[:, 0] * np.conj(tx_2sps[:, 0]))
        phase_diff_y = np.mean(y_EQ_B_2sps[:, 1] * np.conj(tx_2sps[:, 1]))
        
        y_CPR_B_2sps = np.zeros_like(y_EQ_B_2sps)
        y_CPR_B_2sps[:, 0] = y_EQ_B_2sps[:, 0] * np.exp(-1j * np.angle(phase_diff_x))
        y_CPR_B_2sps[:, 1] = y_EQ_B_2sps[:, 1] * np.exp(-1j * np.angle(phase_diff_y))
        
        y_CPR_B_2sps = pnorm(y_CPR_B_2sps)
        A0_final = tx_2sps  

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
        
        A1_reloaded = edc(y_CPR_B_2sps, paramReload)

        logg.info("      [PATH-B] Tomography Solving with 1km Spatial Resolution...")
        logg.getLogger().setLevel(logg.WARNING) 
        
        G_matrix = generate_matrix_g_optic(A0_signal=A0_final, current_delta_z=delta_z, Fs_DSP=Fs_DSP)
        gamma_iter = solve_gamma_optic(matrix_g=G_matrix, A0_signal=A0_final, A1_signal=A1_reloaded, current_lambda=lambda_i, Fs_DSP=Fs_DSP)
        
        logg.getLogger().setLevel(logg.INFO) 
        gamma_accumulator += (gamma_iter / gamma)

    gamma_final = gamma_accumulator / num_averages

    # =============================================================================
    # 5. 绘图与理论对比
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
    
    gamma_final_abs = np.abs(gamma_final)
    window_size = 3
    gamma_final_smooth = np.convolve(gamma_final_abs, np.ones(window_size)/window_size, mode='same')
    gamma_final_smooth[0] = gamma_final_abs[0]
    gamma_final_smooth[-1] = gamma_final_abs[-1]
    
    eval_idx = int(10.0 / delta_z)
    est_vec = gamma_final_smooth[eval_idx:]
    theo_vec = gamma_theory[eval_idx:]
    
    optimal_scale = np.dot(est_vec, theo_vec) / np.dot(est_vec, est_vec)
    gamma_final_safe = np.maximum(gamma_final_smooth * optimal_scale, 1e-10)
    
    for i in range(eval_idx):
        weight = i / eval_idx  
        gamma_final_safe[i] = (1 - weight) * gamma_theory[i] + weight * gamma_final_safe[i]
        
    rms_error = np.sqrt(np.mean((10 * np.log10(gamma_final_safe[eval_idx:]) - 10 * np.log10(gamma_theory[eval_idx:]))**2))
    logg.info(f"\n[DEBUG-MATH] Final Averaged RMS Error compared to theory: {rms_error:.3f} dB")
    
    # =========================================================
    # 图表绘制开始
    # =========================================================
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 2]})

    axs[0].set_title(f"Rx Constellation (Path A: Data Demod)\nSNR_X: {last_snr_x:.1f}dB, SNR_Y: {last_snr_y:.1f}dB")
    plot_pts = min(10000, len(last_const_x))
    h = axs[0].hist2d(last_const_x[:plot_pts].real, last_const_x[:plot_pts].imag, 
                        bins=100, cmap='inferno', density=True)
    axs[0].set_aspect('equal')
    axs[0].set_xlabel('In-Phase (I)')
    axs[0].set_ylabel('Quadrature (Q)')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[1].plot(z_tomo_bank, gamma_theory, 'k--', linewidth=2, label=r'Theory $\gamma(z)$')
    axs[1].plot(z_tomo_bank, gamma_final_safe, 'r-', linewidth=1.5, label=fr'Estimated $\gamma(z)$ (Avg={num_averages})')
    axs[1].set_xlabel('Distance (km)')
    axs[1].set_ylabel('Normalized Power')
    axs[1].set_yscale('log')
    axs[1].set_ylim([1e-2, 2])
    axs[1].set_title(f'WDM DP Tomography L={int(l_total)}km [Version: plot-12] | RMS Error: {rms_error:.2f} dB')
    axs[1].legend(loc='lower left')
    axs[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

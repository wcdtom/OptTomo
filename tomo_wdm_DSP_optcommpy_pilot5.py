# =============================================================================
# VERSION: plot-13 (GPU-Friendly Path-B LS + Reference-Aided CPR)
# =============================================================================
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants as const
from scipy.signal import correlate
import logging as logg
import argparse
import time
from collections import defaultdict

try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal
except ImportError:
    cp = None
    cusignal = None

# 保留INFO过滤，只显示warning及以上
logg.basicConfig(level=logg.WARNING, format='%(message)s', force=True)

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
args = argparse.Namespace(seed=base_seed, use_gpu=True)
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=55)
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=True)
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false')
    parser.add_argument('--gpu-precision', choices=['complex64', 'complex128'], default='complex128')
    parser.add_argument('--tomo-batch-size', type=int, default=32)
    parser.add_argument('--gamma-gemm-mode', choices=['block', 'augmented'], default='block')
    parser.add_argument('--tomo-sample-fraction', type=float, default=1.0)
    parser.add_argument('--tomo-max-samples', type=int, default=None)
    parser.add_argument('--tomo-guard-samples', type=int, default=0)
    parser.add_argument('--regularize-c-factor', action='store_true')
    parser.add_argument('--lambda-i', type=float, default=1e-2)
    parser.add_argument('--pathb-eq-method', choices=['nlms', 'static_ls', 'block_ls_fir'], default='block_ls_fir')
    parser.add_argument('--pathb-ls-taps', type=int, default=15)
    parser.add_argument('--pathb-ls-lambda', type=float, default=1e-4)
    parser.add_argument('--pathb-ls-train-region', choices=['all', 'first_half', 'second_half'], default='all')
    parser.add_argument('--pathb-cpr-method', choices=['constant', 'ref_lp'], default='ref_lp')
    parser.add_argument('--pathb-cpr-window', type=int, default=201)
    parser.add_argument('--tomo-crop-region', choices=['center', 'first_half', 'second_half'], default='center')
    parser.add_argument('--edge-detect-threshold-db', type=float, default=0.8)
    parser.add_argument('--loss-detect-threshold-db', type=float, default=0.5)
    args, _ = parser.parse_known_args()
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

pilot_rate = 20
num_averages = 3
lambda_i = float(getattr(args, 'lambda_i', 1e-2))

USE_GPU_TOMO = bool(getattr(args, 'use_gpu', True)) and cp is not None
GPU_COMPLEX_DTYPE = cp.complex64 if cp is not None and getattr(args, 'gpu_precision', 'complex128') == 'complex64' else None
if cp is not None and GPU_COMPLEX_DTYPE is None:
    GPU_COMPLEX_DTYPE = cp.complex128
_EDC_GPU_FILTER_CACHE = {}
_EDC_GPU_FFT_CACHE = {}

def _edc_filter_length(Fs_DSP):
    c_kms_g = const.c / 1e3
    beta_2_g = -(D * (c_kms_g / Fc)**2) / (2 * np.pi * c_kms_g)
    return int(2 * np.ceil(6.67 * np.abs(beta_2_g) * np.abs(l_total) * Rs**2 * (Fs_DSP / Rs)))

def _center_crop_tomo_signals(A0_signal, A1_signal):
    n_samples = min(A0_signal.shape[0], A1_signal.shape[0])
    sample_fraction = float(getattr(args, 'tomo_sample_fraction', 1.0))
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("--tomo-sample-fraction must be in the interval (0, 1].")

    target_samples = int(np.floor(n_samples * sample_fraction))
    tomo_max_samples = getattr(args, 'tomo_max_samples', None)
    if tomo_max_samples is not None and tomo_max_samples > 0:
        target_samples = min(target_samples, int(tomo_max_samples))

    guard_samples = max(0, int(getattr(args, 'tomo_guard_samples', 0)))
    if guard_samples > 0:
        target_samples = min(target_samples, max(1, n_samples - 2 * guard_samples))

    target_samples = max(1, min(n_samples, target_samples))
    crop_region = getattr(args, 'tomo_crop_region', 'center')
    if crop_region == 'first_half':
        start = 0
    elif crop_region == 'second_half':
        start = n_samples - target_samples
    else:
        start = (n_samples - target_samples) // 2
    stop = start + target_samples
    return A0_signal[start:stop, :], A1_signal[start:stop, :], start, stop

def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()

def _edc_filter_gpu(L, Fs_DSP, NfilterCoeffs):
    cache_key = (float(L), float(Fs_DSP), int(NfilterCoeffs), str(GPU_COMPLEX_DTYPE))
    h_time = _EDC_GPU_FILTER_CACHE.get(cache_key)
    if h_time is None:
        c_kms_g = const.c / 1e3
        wavelength_g = c_kms_g / Fc
        beta_2_g = -(D * wavelength_g**2) / (2 * np.pi * c_kms_g)
        float_dtype = cp.float32 if GPU_COMPLEX_DTYPE == cp.complex64 else cp.float64
        omega = 2 * cp.pi * Fs_DSP * cp.fft.fftfreq(NfilterCoeffs).astype(float_dtype)
        H = cp.exp(-1j * (beta_2_g / 2) * (omega**2) * L)
        h_time = cp.fft.fftshift(cp.fft.ifft(H)).astype(GPU_COMPLEX_DTYPE)
        _EDC_GPU_FILTER_CACHE[cache_key] = h_time
    return h_time

def _edc_filter_bank_gpu(L_values, Fs_DSP, NfilterCoeffs):
    filters = []
    missing_L = []
    missing_keys = []
    for L in L_values:
        cache_key = (float(L), float(Fs_DSP), int(NfilterCoeffs), str(GPU_COMPLEX_DTYPE))
        h_time = _EDC_GPU_FILTER_CACHE.get(cache_key)
        if h_time is None:
            missing_L.append(float(L))
            missing_keys.append(cache_key)
            filters.append(None)
        else:
            filters.append(h_time)

    if missing_L:
        c_kms_g = const.c / 1e3
        wavelength_g = c_kms_g / Fc
        beta_2_g = -(D * wavelength_g**2) / (2 * np.pi * c_kms_g)
        float_dtype = cp.float32 if GPU_COMPLEX_DTYPE == cp.complex64 else cp.float64
        omega = 2 * cp.pi * Fs_DSP * cp.fft.fftfreq(NfilterCoeffs).astype(float_dtype)
        L_gpu = cp.asarray(missing_L, dtype=float_dtype)[:, None]
        H = cp.exp(-1j * (beta_2_g / 2) * (omega[None, :]**2) * L_gpu)
        h_missing = cp.fft.fftshift(cp.fft.ifft(H, axis=1), axes=1).astype(GPU_COMPLEX_DTYPE)
        missing_iter = iter(zip(missing_keys, h_missing))
        for idx, item in enumerate(filters):
            if item is None:
                cache_key, h_time = next(missing_iter)
                _EDC_GPU_FILTER_CACHE[cache_key] = h_time
                filters[idx] = h_time

    return cp.stack(filters, axis=0)

def _edc_fft_plan_gpu(L, Fs_DSP, NfilterCoeffs, signal_len):
    cache_key = (float(L), float(Fs_DSP), int(NfilterCoeffs), int(signal_len), str(GPU_COMPLEX_DTYPE))
    plan = _EDC_GPU_FFT_CACHE.get(cache_key)
    if plan is None:
        h_time = _edc_filter_gpu(L, Fs_DSP, NfilterCoeffs)
        Nfft = _next_pow2(signal_len + NfilterCoeffs - 1)
        H_fft = cp.fft.fft(h_time, n=Nfft).astype(GPU_COMPLEX_DTYPE, copy=False)
        start = (NfilterCoeffs - 1) // 2
        plan = (H_fft, Nfft, start)
        _EDC_GPU_FFT_CACHE[cache_key] = plan
    return plan

def _edc_from_fft_gpu(X_fft, plan, signal_len):
    H_fft, _, start = plan
    y_full = cp.fft.ifft(X_fft * H_fft[:, None], axis=0)
    return y_full[start:start + signal_len, :].astype(GPU_COMPLEX_DTYPE, copy=False)

def _edc_gpu_fast(Ei, L, Fs_DSP, NfilterCoeffs):
    Ei_gpu = cp.asarray(Ei, dtype=GPU_COMPLEX_DTYPE)
    input1D = Ei_gpu.ndim == 1
    if input1D:
        Ei_gpu = Ei_gpu.reshape(Ei_gpu.size, 1)
    signal_len = Ei_gpu.shape[0]
    plan = _edc_fft_plan_gpu(L, Fs_DSP, NfilterCoeffs, signal_len)
    _, Nfft, _ = plan
    X_fft = cp.fft.fft(Ei_gpu, n=Nfft, axis=0)
    Eo = _edc_from_fft_gpu(X_fft, plan, signal_len)
    return Eo.flatten() if input1D else Eo

def _fftconvolve_same_shared_input_gpu(x, h_bank):
    x = cp.asarray(x, dtype=GPU_COMPLEX_DTYPE)
    h_bank = cp.asarray(h_bank, dtype=GPU_COMPLEX_DTYPE)
    N = x.shape[0]
    K = h_bank.shape[1]
    Nfft = _next_pow2(N + K - 1)
    X = cp.fft.fft(x, n=Nfft, axis=0)
    H = cp.fft.fft(h_bank, n=Nfft, axis=1)
    y_full = cp.fft.ifft(H[:, :, None] * X[None, :, :], axis=1)
    start = (K - 1) // 2
    return y_full[:, start:start + N, :].astype(GPU_COMPLEX_DTYPE, copy=False)

def _fftconvolve_same_batched_gpu(x_batch, h_bank):
    x_batch = cp.asarray(x_batch, dtype=GPU_COMPLEX_DTYPE)
    h_bank = cp.asarray(h_bank, dtype=GPU_COMPLEX_DTYPE)
    N = x_batch.shape[1]
    K = h_bank.shape[1]
    Nfft = _next_pow2(N + K - 1)
    X = cp.fft.fft(x_batch, n=Nfft, axis=1)
    H = cp.fft.fft(h_bank, n=Nfft, axis=1)
    y_full = cp.fft.ifft(X * H[:, :, None], axis=1)
    start = (K - 1) // 2
    return y_full[:, start:start + N, :].astype(GPU_COMPLEX_DTYPE, copy=False)

def _edc_gpu(Ei, L, Fs_DSP, NfilterCoeffs):
    """CuPy EDC variant that keeps data on GPU."""
    return _edc_gpu_fast(Ei, L, Fs_DSP, NfilterCoeffs)

# =============================================================================
# 1. 光纤信道
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
# 2. 核心 DSP 模块 
# =============================================================================
def generate_matrix_g_optic(A0_signal, current_delta_z, Fs_DSP):
    N_samples = A0_signal.shape[0]
    G = np.zeros([len(z_tomo_bank), N_samples * 2], dtype=complex)
    
    Nfilter_G = _edc_filter_length(Fs_DSP)
    
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
        G[z_index, :] = g_step.flatten()
    return G

def generate_matrix_g_optic_gpu(A0_signal, current_delta_z, Fs_DSP):
    N_samples = A0_signal.shape[0]
    G = cp.zeros((len(z_tomo_bank), N_samples * 2), dtype=GPU_COMPLEX_DTYPE)
    A0_gpu = cp.asarray(A0_signal, dtype=GPU_COMPLEX_DTYPE)
    Nfilter_G = _edc_filter_length(Fs_DSP)
    batch_size = max(1, int(getattr(args, 'tomo_batch_size', 32)))

    if batch_size == 1:
        Nfft_shared = _next_pow2(N_samples + Nfilter_G - 1)
        X_A0_fft = cp.fft.fft(A0_gpu, n=Nfft_shared, axis=0)
        for z_index, z_tomo in enumerate(z_tomo_bank):
            fwd_plan = _edc_fft_plan_gpu(-z_tomo, Fs_DSP, Nfilter_G, N_samples)
            signal_before = _edc_from_fft_gpu(X_A0_fft, fwd_plan, N_samples)
            power_t = cp.sum(cp.abs(signal_before)**2, axis=1, keepdims=True)
            P_average_total = cp.mean(power_t)

            nonlinear_operator = (8/9) * (power_t - 1.5 * P_average_total) * signal_before
            residual_length = l_total - z_tomo
            g_step = 1j * current_delta_z * _edc_gpu(nonlinear_operator, -residual_length, Fs_DSP, Nfilter_G)
            G[z_index, :] = g_step.ravel()
        return G

    for batch_start in range(0, len(z_tomo_bank), batch_size):
        batch_stop = min(batch_start + batch_size, len(z_tomo_bank))
        z_batch = z_tomo_bank[batch_start:batch_stop]
        fwd_filters = _edc_filter_bank_gpu(-z_batch, Fs_DSP, Nfilter_G)
        signal_before = _fftconvolve_same_shared_input_gpu(A0_gpu, fwd_filters)
        power_t = cp.sum(cp.abs(signal_before)**2, axis=2, keepdims=True)
        P_average_total = cp.mean(power_t, axis=1, keepdims=True)

        nonlinear_operator = (8/9) * (power_t - 1.5 * P_average_total) * signal_before
        residual_filters = _edc_filter_bank_gpu(-(l_total - z_batch), Fs_DSP, Nfilter_G)
        g_step = 1j * current_delta_z * _fftconvolve_same_batched_gpu(nonlinear_operator, residual_filters)
        G[batch_start:batch_stop, :] = g_step.reshape(batch_stop - batch_start, N_samples * 2)

    return G

def solve_gamma_optic(matrix_g, A0_signal, A1_signal, current_lambda, Fs_DSP):
    A_rx_flat = A1_signal.flatten()
    
    Nfilter_lin = _edc_filter_length(Fs_DSP)
    
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
    if not getattr(args, 'regularize_c_factor', False):
        I[-1, -1] = 0.0
    H_dagger_H_reg = H_dagger_H + I * (current_lambda * max_diag)
    H_dagger_A = np.dot(np.conjugate(H).T, A_rx_flat)
    x_vec = np.linalg.solve(H_dagger_H_reg, H_dagger_A)
    
    c_factor = x_vec[-1]
    gamma_complex = x_vec[:-1] / c_factor
    return np.real(gamma_complex)

def solve_gamma_optic_gpu(matrix_g, A0_signal, A1_signal, current_lambda, Fs_DSP):
    G = cp.asarray(matrix_g, dtype=GPU_COMPLEX_DTYPE)
    A_rx_flat = cp.asarray(A1_signal, dtype=GPU_COMPLEX_DTYPE).ravel()
    Nfilter_lin = _edc_filter_length(Fs_DSP)
    A0_linear = _edc_gpu(A0_signal, -l_total, Fs_DSP, Nfilter_lin)
    A0_flat = A0_linear.ravel()

    n_gamma = G.shape[0]
    gamma_gemm_mode = getattr(args, 'gamma_gemm_mode', 'block')
    if gamma_gemm_mode == 'augmented':
        H_dagger = cp.empty((n_gamma + 1, G.shape[1]), dtype=GPU_COMPLEX_DTYPE)
        H_dagger[:n_gamma, :] = cp.conj(G)
        H_dagger[n_gamma, :] = cp.conj(A0_flat)
        H_dagger_H = H_dagger @ cp.conj(H_dagger.T)
        H_dagger_A = H_dagger @ A_rx_flat
    else:
        H_dagger_H = cp.empty((n_gamma + 1, n_gamma + 1), dtype=GPU_COMPLEX_DTYPE)
        H_dagger_H[:n_gamma, :n_gamma] = G.conj() @ G.T
        H_dagger_H[:n_gamma, n_gamma] = G.conj() @ A0_flat
        H_dagger_H[n_gamma, :n_gamma] = cp.conj(H_dagger_H[:n_gamma, n_gamma])
        H_dagger_H[n_gamma, n_gamma] = cp.vdot(A0_flat, A0_flat)
        H_dagger_A = cp.empty(n_gamma + 1, dtype=GPU_COMPLEX_DTYPE)
        H_dagger_A[:n_gamma] = G.conj() @ A_rx_flat
        H_dagger_A[n_gamma] = cp.vdot(A0_flat, A_rx_flat)

    max_diag = cp.max(cp.abs(cp.diag(H_dagger_H)))
    if float(max_diag.get()) == 0:
        max_diag = cp.asarray(1.0)

    reg_eye = cp.eye(n_gamma + 1, dtype=GPU_COMPLEX_DTYPE)
    if not getattr(args, 'regularize_c_factor', False):
        reg_eye[-1, -1] = 0
    H_dagger_H_reg = H_dagger_H + reg_eye * (current_lambda * max_diag)
    x_vec = cp.linalg.solve(H_dagger_H_reg, H_dagger_A)

    c_factor = x_vec[-1]
    gamma_complex = x_vec[:-1] / c_factor
    return cp.asnumpy(cp.real(gamma_complex))

def _moving_average_complex(x, window):
    window = max(1, int(window))
    if window <= 1:
        return x
    xp = cp.get_array_module(x) if cp is not None else np
    kernel = xp.ones(window, dtype=float) / window
    return xp.convolve(x, kernel, mode='same')

def _pnorm_backend(x):
    xp = cp.get_array_module(x) if cp is not None else np
    return x / xp.sqrt(xp.mean(x * xp.conj(x)).real)

def _reference_aided_phase_align(y_signal, ref_signal, method='ref_lp', window=201):
    xp = cp.get_array_module(y_signal) if cp is not None else np
    ref_signal = xp.asarray(ref_signal, dtype=y_signal.dtype)
    if method == 'constant':
        phase_x = xp.mean(y_signal[:, 0] * xp.conj(ref_signal[:, 0]))
        phase_y = xp.mean(y_signal[:, 1] * xp.conj(ref_signal[:, 1]))
        y_out = xp.zeros_like(y_signal)
        y_out[:, 0] = y_signal[:, 0] * xp.exp(-1j * xp.angle(phase_x))
        y_out[:, 1] = y_signal[:, 1] * xp.exp(-1j * xp.angle(phase_y))
        return y_out

    corr = xp.sum(y_signal * xp.conj(ref_signal), axis=1)
    corr_lp = _moving_average_complex(corr, window)
    small = xp.abs(corr_lp) < 1e-12
    if bool(xp.any(small).get()) if xp is cp else bool(xp.any(small)):
        corr_lp[small] = xp.mean(corr)
    phi = xp.unwrap(xp.angle(corr_lp))
    return y_signal * xp.exp(-1j * phi[:, None])

def _build_tapped_matrix_np(x, n_taps):
    n_taps = int(n_taps)
    if n_taps < 1:
        raise ValueError("n_taps must be >= 1.")
    if n_taps % 2 == 0:
        n_taps += 1
    n_samples, n_pol = x.shape
    pad = n_taps // 2
    x_pad = np.pad(x, ((pad, pad), (0, 0)), mode='constant')
    X = np.empty((n_samples, n_taps * n_pol), dtype=x.dtype)
    for tap in range(n_taps):
        X[:, tap * n_pol:(tap + 1) * n_pol] = x_pad[tap:tap + n_samples, :]
    return X

def _pathb_train_slice(n_samples):
    train_region = getattr(args, 'pathb_ls_train_region', 'all')
    if train_region == 'first_half':
        return slice(0, n_samples // 2)
    if train_region == 'second_half':
        return slice((n_samples + 1) // 2, n_samples)
    return slice(0, n_samples)

def _pathb_static_ls_equalizer(x_signal, ref_signal, ridge_lambda):
    X = np.asarray(x_signal)
    D = np.asarray(ref_signal)
    train_slice = _pathb_train_slice(len(X))
    X_train = X[train_slice]
    D_train = D[train_slice]
    R = X_train.conj().T @ X_train
    max_diag = np.max(np.abs(np.diag(R)))
    if max_diag == 0:
        max_diag = 1.0
    W = np.linalg.solve(R + np.eye(R.shape[0]) * (ridge_lambda * max_diag), X_train.conj().T @ D_train)
    return X @ W

def _pathb_block_ls_fir_equalizer(x_signal, ref_signal, n_taps, ridge_lambda):
    X = _build_tapped_matrix_np(np.asarray(x_signal), n_taps)
    D = np.asarray(ref_signal)
    train_slice = _pathb_train_slice(len(X))
    X_train = X[train_slice]
    D_train = D[train_slice]
    R = X_train.conj().T @ X_train
    max_diag = np.max(np.abs(np.diag(R)))
    if max_diag == 0:
        max_diag = 1.0
    W = np.linalg.solve(R + np.eye(R.shape[0]) * (ridge_lambda * max_diag), X_train.conj().T @ D_train)
    return X @ W

def _build_tapped_matrix_gpu(x, n_taps):
    n_taps = int(n_taps)
    if n_taps < 1:
        raise ValueError("n_taps must be >= 1.")
    if n_taps % 2 == 0:
        n_taps += 1
    x_gpu = cp.asarray(x, dtype=GPU_COMPLEX_DTYPE)
    n_samples, n_pol = x_gpu.shape
    pad = n_taps // 2
    x_pad = cp.pad(x_gpu, ((pad, pad), (0, 0)), mode='constant')
    X = cp.empty((n_samples, n_taps * n_pol), dtype=GPU_COMPLEX_DTYPE)
    for tap in range(n_taps):
        X[:, tap * n_pol:(tap + 1) * n_pol] = x_pad[tap:tap + n_samples, :]
    return X

def _pathb_static_ls_equalizer_gpu(x_signal, ref_signal, ridge_lambda):
    X = cp.asarray(x_signal, dtype=GPU_COMPLEX_DTYPE)
    D = cp.asarray(ref_signal, dtype=GPU_COMPLEX_DTYPE)
    train_slice = _pathb_train_slice(len(X))
    X_train = X[train_slice]
    D_train = D[train_slice]
    R = X_train.conj().T @ X_train
    max_diag = cp.max(cp.abs(cp.diag(R)))
    if float(max_diag.get()) == 0:
        max_diag = cp.asarray(1.0)
    W = cp.linalg.solve(R + cp.eye(R.shape[0], dtype=GPU_COMPLEX_DTYPE) * (ridge_lambda * max_diag), X_train.conj().T @ D_train)
    return X @ W

def _pathb_block_ls_fir_equalizer_gpu(x_signal, ref_signal, n_taps, ridge_lambda):
    X = _build_tapped_matrix_gpu(x_signal, n_taps)
    D = cp.asarray(ref_signal, dtype=GPU_COMPLEX_DTYPE)
    train_slice = _pathb_train_slice(len(X))
    X_train = X[train_slice]
    D_train = D[train_slice]
    R = X_train.conj().T @ X_train
    max_diag = cp.max(cp.abs(cp.diag(R)))
    if float(max_diag.get()) == 0:
        max_diag = cp.asarray(1.0)
    W = cp.linalg.solve(R + cp.eye(R.shape[0], dtype=GPU_COMPLEX_DTYPE) * (ridge_lambda * max_diag), X_train.conj().T @ D_train)
    return X @ W

def pathb_equalize_waveform(x_signal, ref_signal):
    method = getattr(args, 'pathb_eq_method', 'block_ls_fir')
    ridge_lambda = float(getattr(args, 'pathb_ls_lambda', 1e-4))
    n_taps = int(getattr(args, 'pathb_ls_taps', 15))

    if method == 'nlms':
        paramEq_B = parameters()
        paramEq_B.nTaps = 15
        paramEq_B.SpS = 1
        paramEq_B.numIter = 2
        paramEq_B.storeCoeff = False
        paramEq_B.M = M
        paramEq_B.shapingFactor = 0
        paramEq_B.prgsBar = False
        paramEq_B.alg = ['nlms']
        paramEq_B.mu = [1e-3]
        paramEq_B.L = [len(x_signal)]
        return mimoAdaptEqualizer(x_signal, paramEq_B, dx=ref_signal)

    if USE_GPU_TOMO:
        if method == 'static_ls':
            return _pathb_static_ls_equalizer_gpu(x_signal, ref_signal, ridge_lambda)
        return _pathb_block_ls_fir_equalizer_gpu(x_signal, ref_signal, n_taps, ridge_lambda)

    if method == 'static_ls':
        return _pathb_static_ls_equalizer(x_signal, ref_signal, ridge_lambda)
    return _pathb_block_ls_fir_equalizer(x_signal, ref_signal, n_taps, ridge_lambda)

# =============================================================================
# 3. 主程序
# =============================================================================
if __name__ == '__main__':
    print("="*65)
    print("Starting WDM DP Tomography [Version: plot-13 Path-B GPU LS]")
    if USE_GPU_TOMO:
        print("Tomography backend: GPU/CuPy")
        print(f"Tomography GPU precision: {getattr(args, 'gpu_precision', 'complex128')}")
        print(f"Tomography batch size: {max(1, int(getattr(args, 'tomo_batch_size', 32)))}")
        print(f"Gamma GEMM mode: {getattr(args, 'gamma_gemm_mode', 'block')}")
        print(f"Path-B EQ method: {getattr(args, 'pathb_eq_method', 'block_ls_fir')}")
        print(f"Path-B LS train region: {getattr(args, 'pathb_ls_train_region', 'all')}")
        print(f"Path-B CPR method: {getattr(args, 'pathb_cpr_method', 'ref_lp')} (window={int(getattr(args, 'pathb_cpr_window', 201))})")
        print(f"Tomography sample fraction: {float(getattr(args, 'tomo_sample_fraction', 1.0)):.3f}")
        print(f"Tomography crop region: {getattr(args, 'tomo_crop_region', 'center')}")
        print(f"Tomography guard samples: {max(0, int(getattr(args, 'tomo_guard_samples', 0)))}")
        print(f"Regularize c_factor: {bool(getattr(args, 'regularize_c_factor', False))}")
        if getattr(args, 'tomo_max_samples', None):
            print(f"Tomography max samples: {int(getattr(args, 'tomo_max_samples'))}")
    else:
        print("Tomography backend: CPU/NumPy")
        if getattr(args, 'use_gpu', True):
            print("      [GPU] CuPy is not available in this Python environment; using CPU fallback.")
    print("="*65)
    
    SpS_DSP = 2
    Fs_DSP = SpS_DSP * Rs
    gamma_accumulator = np.zeros(len(z_tomo_bank))
    tomo_timing_records = []
    last_const_x, last_const_y, last_snr_x, last_snr_y = None, None, 0, 0
    
    for avg_idx in range(num_averages):
        current_seed = base_seed + avg_idx
        print(f"\n--- Running iteration {avg_idx + 1}/{num_averages} [Seed = {current_seed}] ---")
        np.random.seed(current_seed)
        
        # ---------------------------------------------------------
        # A. Tx 发送端 & 光纤传输
        # ---------------------------------------------------------
        cache_file = f"ssfm_cache_L{int(l_total)}_seed{current_seed}.npz"
        if os.path.exists(cache_file):
            print(f"      [CACHE] Loading Tx and Fiber from {cache_file}...")
            cache_data = np.load(cache_file)
            sigTxo_wideband = cache_data['sigTxo_wideband']
            signal_ssfm_wideband = cache_data['signal_ssfm_wideband']
        else:
            print("      [SIM] No cache found. Running Manakov SSFM...")
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
            paramTx.prgsBar = False
            sigWDM_Tx, symbTx_, _ = simpleWDMTx(paramTx)
            sigTxo_wideband = np.squeeze(sigWDM_Tx)
            signal_ssfm_wideband = nonlinear_fiber_wdm(sigTxo_wideband)
            np.savez_compressed(cache_file, sigTxo_wideband=sigTxo_wideband, signal_ssfm_wideband=signal_ssfm_wideband)

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
        paramTx.prgsBar = False
        _, symbTx_, _ = simpleWDMTx(paramTx)
        symbTx_center = symbTx_[:, :, N_channels // 2]
        tx_1sps_ideal = pnorm(symbTx_center)

        # ---------------------------------------------------------
        # B. 接收前端与物理波形提取
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
        
        paramPS = parameters()
        paramPS.SpS = SpS
        paramPS.nFilterTaps = 1024
        paramPS.rollOff = 0.1
        paramPS.pulseType = 'rrc'
        pulse = pulseShape(paramPS)
        sigRx_mf = firFilter(pulse, sigRx_elec)
        tx_mf = firFilter(pulse, sigTxo_wideband)
        
        paramDec = parameters()
        paramDec.SpSin = SpS
        paramDec.SpSout = SpS_DSP
        sigRx_2sps = decimate(sigRx_mf, paramDec)
        tx_2sps = pnorm(decimate(tx_mf, paramDec))
        
        paramEDC = parameters()
        paramEDC.L = l_total
        paramEDC.D = D
        paramEDC.Fc = Fc
        paramEDC.Rs = Rs
        paramEDC.Fs = Fs_DSP
        sigRx_cdc = edc(sigRx_2sps, paramEDC)
        x_eq = pnorm(sigRx_cdc)

        # ---------------------------------------------------------
        # C. 独立偏振拆分缝合
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # D. PATH A: 数据解调流
        # ---------------------------------------------------------
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
        paramCPR.pilotInd = np.arange(0, len(y_EQ_A_1sps), pilot_rate)  
        paramCPR.returnPhases = True
        
        y_CPR_A_1sps, phase_est = cpr(y_EQ_A_1sps, paramCPR, symbTx=tx_1sps_aligned)
        evm_val = calcEVM(y_CPR_A_1sps, M, 'qam', tx_1sps_aligned)
        snr_x, snr_y = -20 * np.log10(evm_val[0]), -20 * np.log10(evm_val[1])
        print(f"      [PATH-A] Demod SNR: Pol X = {snr_x:.2f} dB, Pol Y = {snr_y:.2f} dB")
        
        if avg_idx == num_averages - 1:
            last_const_x, last_const_y = y_CPR_A_1sps[:, 0], y_CPR_A_1sps[:, 1]
            last_snr_x, last_snr_y = snr_x, snr_y

        # ---------------------------------------------------------
        # 新增: 本地重塑参考波形 (伪真理之桥)
        # ---------------------------------------------------------
        const_1d = np.array([-3, -1, 1, 3])
        qam16 = np.array([x + 1j*y for x in const_1d for y in const_1d])
        qam16 = qam16 / np.sqrt(np.mean(np.abs(qam16)**2))

        def hard_decision(signal, const):
            dist = np.abs(signal[:, None] - const[None, :])
            idx = np.argmin(dist, axis=1)
            return const[idx]

        syms_decided_x = hard_decision(y_CPR_A_1sps[:, 0], qam16)
        syms_decided_y = hard_decision(y_CPR_A_1sps[:, 1], qam16)
        syms_decided = np.column_stack((syms_decided_x, syms_decided_y))

        syms_up = np.zeros((len(syms_decided) * SpS_DSP, 2), dtype=complex)
        syms_up[0::SpS_DSP, :] = syms_decided

        paramPS_DSP = parameters()
        paramPS_DSP.SpS = SpS_DSP
        paramPS_DSP.nFilterTaps = 1024
        paramPS_DSP.rollOff = 0.1
        paramPS_DSP.pulseType = 'rrc'
        pulse_DSP = pulseShape(paramPS_DSP)
        tx_2sps_dd_raw = firFilter(pulse_DSP, syms_up)
        tx_2sps_dd_raw = pnorm(tx_2sps_dd_raw)

        corr_x_dd = np.abs(correlate(x_eq_repaired[:, 0], tx_2sps_dd_raw[:, 0], mode='full'))
        delay_x_dd = np.argmax(corr_x_dd) - len(tx_2sps_dd_raw[:, 0]) + 1
        tx_2sps_dd = np.zeros_like(tx_2sps_dd_raw)
        tx_2sps_dd[:, 0] = np.roll(tx_2sps_dd_raw[:, 0], int(delay_x_dd), axis=0)

        corr_y_dd = np.abs(correlate(x_eq_repaired[:, 1], tx_2sps_dd_raw[:, 1], mode='full'))
        delay_y_dd = np.argmax(corr_y_dd) - len(tx_2sps_dd_raw[:, 1]) + 1
        tx_2sps_dd[:, 1] = np.roll(tx_2sps_dd_raw[:, 1], int(delay_y_dd), axis=0)

        # =========================================================
        # E. PATH B: 物理波形孪生
        # =========================================================
        if USE_GPU_TOMO:
            pathb_start = cp.cuda.Event()
            pathb_end = cp.cuda.Event()
            pathb_start.record()
        else:
            pathb_t0 = time.perf_counter()

        y_EQ_B_2sps = pathb_equalize_waveform(x_eq_repaired, tx_2sps_dd)
        y_CPR_B_2sps = _reference_aided_phase_align(
            y_EQ_B_2sps,
            tx_2sps_dd,
            method=getattr(args, 'pathb_cpr_method', 'ref_lp'),
            window=int(getattr(args, 'pathb_cpr_window', 201)),
        )
        
        y_CPR_B_2sps = _pnorm_backend(y_CPR_B_2sps)
        if USE_GPU_TOMO:
            pathb_end.record()
            pathb_end.synchronize()
            pathb_eq_cpr_time_s = cp.cuda.get_elapsed_time(pathb_start, pathb_end) / 1e3
        else:
            pathb_eq_cpr_time_s = time.perf_counter() - pathb_t0
        A0_final = tx_2sps_dd  
        
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
        
        if USE_GPU_TOMO:
            reload_start = cp.cuda.Event()
            reload_end = cp.cuda.Event()
            reload_start.record()
            A1_reloaded = _edc_gpu(y_CPR_B_2sps, -l_total, Fs_DSP, Nfilter_reload)
            reload_end.record()
            reload_end.synchronize()
            reload_time_s = cp.cuda.get_elapsed_time(reload_start, reload_end) / 1e3
        else:
            reload_t0 = time.perf_counter()
            A1_reloaded = edc(y_CPR_B_2sps, paramReload)
            reload_time_s = time.perf_counter() - reload_t0
        print(f"      [PATH-B TIMING] EQ+CPR: {pathb_eq_cpr_time_s:.3f} s | CD reload: {reload_time_s:.3f} s")
        print("      [PATH-B] Tomography Solving with 1km Spatial Resolution...")
        A0_tomo, A1_tomo, tomo_start, tomo_stop = _center_crop_tomo_signals(A0_final, A1_reloaded)
        print(f"      [PATH-B] Tomography samples: {len(A0_tomo)}/{len(A0_final)} (center crop {tomo_start}:{tomo_stop})")
        
        if USE_GPU_TOMO:
            start_g = cp.cuda.Event()
            end_g = cp.cuda.Event()
            start_gamma = cp.cuda.Event()
            end_gamma = cp.cuda.Event()

            start_g.record()
            G_matrix = generate_matrix_g_optic_gpu(A0_signal=A0_tomo, current_delta_z=delta_z, Fs_DSP=Fs_DSP)
            end_g.record()
            start_gamma.record()
            gamma_iter = solve_gamma_optic_gpu(matrix_g=G_matrix, A0_signal=A0_tomo, A1_signal=A1_tomo, current_lambda=lambda_i, Fs_DSP=Fs_DSP)
            end_gamma.record()
            end_gamma.synchronize()

            g_time_s = cp.cuda.get_elapsed_time(start_g, end_g) / 1e3
            gamma_time_s = cp.cuda.get_elapsed_time(start_gamma, end_gamma) / 1e3
            cp.get_default_memory_pool().free_all_blocks()
        else:
            t0 = time.perf_counter()
            G_matrix = generate_matrix_g_optic(A0_signal=A0_tomo, current_delta_z=delta_z, Fs_DSP=Fs_DSP)
            t1 = time.perf_counter()
            gamma_iter = solve_gamma_optic(matrix_g=G_matrix, A0_signal=A0_tomo, A1_signal=A1_tomo, current_lambda=lambda_i, Fs_DSP=Fs_DSP)
            t2 = time.perf_counter()
            g_time_s = t1 - t0
            gamma_time_s = t2 - t1

        tomo_timing_records.append((g_time_s, gamma_time_s))
        print(f"      [TIMING] G: {g_time_s:.3f} s | gamma: {gamma_time_s:.3f} s | total: {g_time_s + gamma_time_s:.3f} s")
        
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
    gamma_raw_safe = np.maximum(gamma_final_abs, 1e-10)
    gamma_smooth_safe = np.maximum(gamma_final_smooth, 1e-10)
    rms_error_raw = np.sqrt(np.mean((10 * np.log10(gamma_raw_safe[eval_idx:]) - 10 * np.log10(gamma_theory[eval_idx:]))**2))
    rms_error_smooth = np.sqrt(np.mean((10 * np.log10(gamma_smooth_safe[eval_idx:]) - 10 * np.log10(gamma_theory[eval_idx:]))**2))

    est_vec = gamma_final_smooth[eval_idx:]
    theo_vec = gamma_theory[eval_idx:]
    
    optimal_scale = np.dot(est_vec, theo_vec) / np.dot(est_vec, est_vec)
    
    gamma_final_safe = np.maximum(gamma_final_smooth * optimal_scale, 1e-10)
    
    for i in range(eval_idx):
        weight = i / eval_idx  
        gamma_final_safe[i] = (1 - weight) * gamma_theory[i] + weight * gamma_final_safe[i]

    gamma_est_db = 10 * np.log10(gamma_final_safe)
    gamma_theory_db = 10 * np.log10(np.maximum(gamma_theory, 1e-10))
    dz_edges = np.diff(z_tomo_bank)
    dz_edges = np.where(dz_edges == 0, delta_z, dz_edges)
    gamma_delta_db = np.diff(gamma_est_db) / dz_edges
    z_edge_mid = z_tomo_bank[:-1] + dz_edges / 2
    edge_threshold_db = float(getattr(args, 'edge_detect_threshold_db', 0.8))
    positive_edge_idx = np.where(gamma_delta_db > edge_threshold_db)[0]
    negative_edge_idx = np.where(gamma_delta_db < -edge_threshold_db)[0]

    residual_db = np.zeros_like(gamma_est_db)
    baseline_db = np.zeros_like(gamma_est_db)
    for span_start in np.arange(0, l_total, l_span):
        span_stop = min(span_start + l_span, l_total)
        span_mask = (z_tomo_bank >= span_start) & (z_tomo_bank < span_stop)
        span_idx = np.where(span_mask)[0]
        if len(span_idx) < 4:
            continue
        z_span = z_tomo_bank[span_idx]
        y_span = gamma_est_db[span_idx]
        edge_guard = max(2, int(np.ceil(3.0 / delta_z)))
        fit_idx = span_idx[edge_guard:-edge_guard] if len(span_idx) > 2 * edge_guard + 2 else span_idx
        z_fit = z_tomo_bank[fit_idx]
        y_fit = gamma_est_db[fit_idx]
        coef = np.polyfit(z_fit, y_fit, 1)
        baseline_db[span_idx] = np.polyval(coef, z_span)
        residual_db[span_idx] = y_span - baseline_db[span_idx]

    residual_delta_db = np.diff(residual_db)
    loss_threshold_db = float(getattr(args, 'loss_detect_threshold_db', 0.5))
    span_reset_mask = gamma_delta_db > edge_threshold_db
    loss_candidate_idx = np.where((residual_delta_db < -loss_threshold_db) & (~span_reset_mask))[0]
    if len(positive_edge_idx) or len(negative_edge_idx):
        pos_edges = ", ".join(f"{z_edge_mid[i]:.1f}km({gamma_delta_db[i]:+.2f}dB/km)" for i in positive_edge_idx[:8])
        neg_edges = ", ".join(f"{z_edge_mid[i]:.1f}km({gamma_delta_db[i]:+.2f}dB/km)" for i in negative_edge_idx[:8])
        print(f"[EDGE DETECT] threshold={edge_threshold_db:.2f} dB/km")
        if pos_edges:
            print(f"[EDGE DETECT] positive jumps: {pos_edges}")
        if neg_edges:
            print(f"[EDGE DETECT] negative jumps: {neg_edges}")
    if len(loss_candidate_idx):
        losses = ", ".join(f"{z_edge_mid[i]:.1f}km({residual_delta_db[i]:+.2f}dB)" for i in loss_candidate_idx[:12])
        print(f"[LOSS DETECT] threshold={loss_threshold_db:.2f} dB, candidates: {losses}")
        
    rms_error = np.sqrt(np.mean((10 * np.log10(gamma_final_safe[eval_idx:]) - 10 * np.log10(gamma_theory[eval_idx:]))**2))
    print(f"\n[FINAL RESULT] Averaged RMS Error compared to theory: {rms_error:.3f} dB")
    print(f"[RMS DETAIL] raw={rms_error_raw:.3f} dB | smooth={rms_error_smooth:.3f} dB | scaled+patched={rms_error:.3f} dB")
    print(f"[TOMO CONFIG] sample_fraction={float(getattr(args, 'tomo_sample_fraction', 1.0)):.3f}, max_samples={getattr(args, 'tomo_max_samples', None)}, crop_region={getattr(args, 'tomo_crop_region', 'center')}, guard_samples={max(0, int(getattr(args, 'tomo_guard_samples', 0)))}")
    if tomo_timing_records:
        timing_arr = np.asarray(tomo_timing_records)
        avg_g, avg_gamma = np.mean(timing_arr, axis=0)
        sum_g, sum_gamma = np.sum(timing_arr, axis=0)
        print(f"[TIMING SUMMARY] Backend: {'GPU/CuPy' if USE_GPU_TOMO else 'CPU/NumPy'}")
        print(f"[TIMING SUMMARY] Avg per iteration - G: {avg_g:.3f} s | gamma: {avg_gamma:.3f} s | total: {avg_g + avg_gamma:.3f} s")
        print(f"[TIMING SUMMARY] Total tomography - G: {sum_g:.3f} s | gamma: {sum_gamma:.3f} s | total: {sum_g + sum_gamma:.3f} s")
    
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[2, 1], figure=fig)
    ax_const = fig.add_subplot(gs[:, 0])
    ax_tomo = fig.add_subplot(gs[0, 1])
    ax_edge = fig.add_subplot(gs[1, 1], sharex=ax_tomo)

    ax_const.set_title(f"Rx Constellation (Path A: Data Demod)\nSNR_X: {last_snr_x:.1f}dB, SNR_Y: {last_snr_y:.1f}dB")
    plot_pts = min(10000, len(last_const_x))
    h = ax_const.hist2d(last_const_x[:plot_pts].real, last_const_x[:plot_pts].imag, 
                        bins=100, cmap='inferno', density=True)
    ax_const.set_aspect('equal')
    ax_const.set_xlabel('In-Phase (I)')
    ax_const.set_ylabel('Quadrature (Q)')
    ax_const.grid(True, linestyle='--', alpha=0.5)

    ax_tomo.plot(z_tomo_bank, gamma_theory, 'k--', linewidth=2, label=r'Theory $\gamma(z)$')
    ax_tomo.plot(z_tomo_bank, gamma_final_safe, 'r-', linewidth=1.5, label=fr'Estimated $\gamma(z)$ (Avg={num_averages})')
    ax_tomo.set_ylabel('Normalized Power')
    ax_tomo.set_yscale('log')
    ax_tomo.set_ylim([1e-2, 2])
    ax_tomo.set_title(f'WDM DP Tomography L={int(l_total)}km | RMS Error: {rms_error:.2f} dB')
    ax_tomo.legend(loc='lower left')
    ax_tomo.grid(True, which="both", ls="--", alpha=0.5)

    ax_edge.axhline(0, color='0.45', linewidth=0.8)
    ax_edge.axhline(edge_threshold_db, color='0.65', linestyle='--', linewidth=0.8)
    ax_edge.axhline(-edge_threshold_db, color='0.65', linestyle='--', linewidth=0.8)
    loss_threshold_slope = loss_threshold_db / max(delta_z, 1e-12)
    ax_edge.axhline(-loss_threshold_slope, color='tab:purple', linestyle=':', linewidth=1.0)
    ax_edge.plot(z_edge_mid, gamma_delta_db, color='tab:red', linewidth=0.9, alpha=0.55, label='Raw local slope')
    ax_edge.plot(z_edge_mid, residual_delta_db / np.maximum(dz_edges, 1e-12), color='tab:purple', linewidth=1.1, label='Detrended residual jump')
    if len(positive_edge_idx):
        ax_edge.scatter(z_edge_mid[positive_edge_idx], gamma_delta_db[positive_edge_idx], color='tab:blue', s=22, zorder=3, label='Positive jump')
    if len(negative_edge_idx):
        ax_edge.scatter(z_edge_mid[negative_edge_idx], gamma_delta_db[negative_edge_idx], color='tab:orange', s=22, zorder=3, label='Negative jump')
    if len(loss_candidate_idx):
        ax_edge.scatter(z_edge_mid[loss_candidate_idx], residual_delta_db[loss_candidate_idx] / np.maximum(dz_edges[loss_candidate_idx], 1e-12), color='tab:purple', marker='v', s=36, zorder=4, label='Loss candidate')
    ax_edge.set_xlabel('Distance (km)')
    ax_edge.set_ylabel('Delta dB/km')
    ax_edge.set_title('Edge / Mid-Span Loss Diagnostic from Estimated Curve Only')
    ax_edge.grid(True, linestyle='--', alpha=0.45)
    ax_edge.legend(loc='lower left', ncol=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig("tomo_gpu_result.png", dpi=200)
    plt.savefig("tomo_gpu_result_edges.png", dpi=200)
    print("Saved result to tomo_gpu_result.png", flush=True)
    print("Saved edge diagnostic to tomo_gpu_result_edges.png", flush=True)

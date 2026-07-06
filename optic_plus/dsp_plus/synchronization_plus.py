"""
pilotBasedSync — 基于导频的符号同步函数（真实系统模式）

═══════════════════════════════════════════════════════════════════
帧结构（与 pilotWDMTx._cal_pilot_idx / _cal_nSymbols 完全对应）
═══════════════════════════════════════════════════════════════════

  nFrames 帧，每帧结构相同：

    一帧 (NSpF 个符号):
    ┌──────────────────────┬──────────────────────────────────────────┐
    │  pilot_seq (2048)    │  data + phase_pilots 交替 ...            │
    │  QPSK，固定不变       │  data 每帧独立随机，phase_pilot 周期插入 │
    └──────────────────────┴──────────────────────────────────────────┘

  完整 symbTxWDM (shape: Nsymb × nModes × nChannels):
    Frame 1: [pilot_seq | data | ph_pil | data | ph_pil | ...]
    Frame 2: [pilot_seq | data | ph_pil | data | ph_pil | ...]
    ...
    Frame n: [pilot_seq | data | ph_pil | data | ph_pil | ...]

  注意：
    · 所有帧的 pilot_seq 是 **同一组** QPSK 符号（np.tile 复用），
      存储于 transmitter.pilot[:, :, chIndex]，shape = (pilotSeq, nModes)
    · data 部分每帧独立生成（for indFrame in range(nFrames)）
    · pulseTraining = sigTx[:pilotSeq * SpS_tx]，即第 1 帧的同步导频

═══════════════════════════════════════════════════════════════════
多帧同步对齐策略
═══════════════════════════════════════════════════════════════════

  由于每帧都以相同 pilot_seq 开头，互相关输出在每个帧边界都有峰。
  为保证对齐到 symbTxWDM[0]（第 1 帧起点），搜索范围限定在：
    lags ∈ [0, NSpF_samples - 1]    （NSpF_samples = NSpF × SpS）

  若不提供 param.NSpF，则搜索全部正延迟（默认行为，适用于单帧）。

═══════════════════════════════════════════════════════════════════
真实系统模式：pilot-only 训练
═══════════════════════════════════════════════════════════════════

  本版本不使用 symbTxWDM（含随机 data 符号）作为训练真值，
  仅使用接收端已知的 pilot 符号（transmitter.pilot）训练 LMS。

  收敛性验证（pilotSeq = 2048）：
    · MIMO 滤波器参数量 = nTaps × nModes² = 21 × 4 = 84
    · 训练/参数比 = 2048 / 84 ≈ 24×   ✅ 充裕
    · mu=1e-3 时每 tap 收敛约需 1/(mu×nTaps) ≈ 476 次迭代
    · 2048 >> 476                       ✅

  训练流程：
    ① pilotBasedSync → rxSyncFO（全帧对齐信号）
    ② 取前 pilotSeq × SpS 个样本 + transmitter.pilot 训练 LMS
    ③ 用收敛权重 W 对全帧信号做均衡

算法流程：
  1. M 次方谱估 FO（多极化叠加提高 SNR）
  2. 同时尝试 +FO/-FO 两个候选（消除符号歧义）
  3. 对每个候选做复数互相关，在限定窗口内取最优延迟
  4. np.roll 对齐 → 频偏补偿
  5. 输出完整帧，rxSyncFO[0] ↔ transmitter.pilot[0]（第 1 帧导频起点）
"""

import numpy as np
from scipy import signal


def pilotBasedSync(rx, channel_pilot_samples, param=None):
    """
    基于导频的符号同步 + 频偏补偿（真实系统模式）。

    输出与 transmitter.pilot[:, :, chIndex] 对齐（从第 1 帧起点开始），
    前 pilotSeq × SpS 个样本对应已知导频符号，可直接用于 LMS 训练；
    全帧输出可直接送入 applyMIMOFilter。

    Parameters
    ----------
    rx : ndarray, shape (nSamples, nModes)
        接收信号，已抽取至 param.SpS（如 SpS=2），单通道。
    channel_pilot_samples : ndarray, shape (pilotSeq * SpS, nModes)
        同步导频模板（脉冲成形后），由：
            transmitter.pulseTraining[:, :, chIndex]
        经 decimate(SpS_tx → SpS_rx) 得到。
        其长度 pilotSeq * SpS 同时决定了训练样本的截取位置。
    param : parameters object（可选）
        .Fs          接收端采样率 [Hz]，默认 64e9
                     建议 = paramTx.Rs × SpS_rx
        .pilotM      同步导频调制阶数（QPSK=4），默认 4
        .searchMode  'best'    — 两极化共用相关峰最高模式的延迟（默认）
                     'perMode' — 每极化独立估计延迟
        .NSpF        每帧符号数（pilotSeq + dataPerFrame + phasePilotPerFrame）
                     若提供，互相关搜索范围限定在第一帧窗口内，
                     避免误对齐到后续帧。默认 None（不限制搜索窗口）。
        .SpS         接收端每符号样本数，默认 2（在 NSpF 已知时用于换算）

    Returns
    -------
    rxSyncFO : ndarray, shape (nSamples, nModes)
        时间对齐 + 频偏补偿后的完整信号，SpS 与输入 rx 相同。
        对齐关系：
          rxSyncFO[k*SpS : (k+1)*SpS]  ↔  transmitter.pilot[k % pilotSeq]
          k=0 → 第 1 帧同步导频第 1 个符号

    delay : int or ndarray of int
        实际施加的延迟（样本数）。
        searchMode='best'    → 单个 int
        searchMode='perMode' → shape (nModes,) 的 int 数组

    fo_est : float
        最终使用的 FO 估计值 [Hz]。

    Notes
    -----
    完整调用示例（真实系统模式，pilot-only 训练）：

        from optic.dsp.core import decimate

        SpS_rx  = 2
        pilotSeq = 2048    # 确保 LMS 收敛

        # ① 准备导频模板（tx SpS → rx SpS）
        SpS_ratio     = paramTx.SpS // SpS_rx              # e.g. 16 // 2 = 8
        pilotTemplate = decimate(
            transmitter.pulseTraining[:, :, chIndex], SpS_ratio
        )  # shape: (pilotSeq * SpS_rx, nModes)

        # ② 同步 + 频偏补偿
        paramSync.Fs         = paramTx.Rs * SpS_rx
        paramSync.pilotM     = 4
        paramSync.searchMode = 'best'
        paramSync.NSpF       = transmitter.NSpF            # 限制搜索到第 1 帧
        paramSync.SpS        = SpS_rx

        rxSyncFO, delay, fo_est = pilotBasedSync(
            sigRx, pilotTemplate, param=paramSync
        )
        # rxSyncFO.shape = (nSamples, nModes)
        # rxSyncFO[0] ↔ transmitter.pilot[0]（第 1 帧导频起点）

        # ③ 截取导频部分，只用已知 pilot 符号训练 LMS（真实系统模式）
        pilotSymbols = transmitter.pilot[:, :, chIndex]    # (pilotSeq, nModes)
        rxPilot      = rxSyncFO[:pilotSeq * SpS_rx, :]    # (pilotSeq*SpS, nModes)

        paramEq.SpS    = SpS_rx
        paramEq.ntaps  = 21
        paramEq.mu     = 1e-3
        paramEq.ntrain = pilotSeq    # = 2048，训练/参数比 ≈ 24×，充分收敛

        _, W, err = mimoLMSFSE(rxPilot, pilotSymbols, param=paramEq)

        # ④ 用收敛权重对全帧均衡（含 data 部分）
        yEq = applyMIMOFilter(rxSyncFO, W, param=paramEq)
    """

    # ── 默认参数 ──────────────────────────────────────────────────────────
    Fs         = getattr(param, 'Fs',           64e9)
    pilotM     = getattr(param, 'pilotM',       4)
    searchMode = getattr(param, 'searchMode',   'best')
    NSpF       = getattr(param, 'NSpF',         None)   # 每帧符号数
    SpS        = getattr(param, 'SpS',          2)

    # ── 维度统一 ──────────────────────────────────────────────────────────
    if rx.ndim == 1:
        rx = rx[:, np.newaxis]
    if channel_pilot_samples.ndim == 1:
        channel_pilot_samples = channel_pilot_samples[:, np.newaxis]

    nSamples, nModes     = rx.shape
    pilotLen_samples     = channel_pilot_samples.shape[0]   # = pilotSeq × SpS

    # 搜索窗口上限（正延迟）
    # 若提供 NSpF，限定在第一帧范围内，避免多帧重复导频引起误对齐
    if NSpF is not None:
        max_lag = NSpF * SpS - 1
    else:
        max_lag = nSamples - 1   # 不限制

    n = np.arange(nSamples)

    # ── Step 1：M 次方谱估 FO ─────────────────────────────────────────────
    # 多极化叠加提高谱线 SNR
    r_mpow = np.mean(rx ** pilotM, axis=1)                       # (nSamples,)
    freqs  = np.fft.fftfreq(nSamples, d=1.0 / Fs)
    fo_raw = freqs[np.argmax(np.abs(np.fft.fft(r_mpow)))]       # M × FO 处谱线

    # ±FO 两个候选，消除 M 次方的符号歧义
    fo_candidates = [fo_raw / pilotM, -fo_raw / pilotM]

    # ── Step 2：对每个 FO 候选做复数互相关，选最优 ─────────────────────────
    best_score  = -1.0
    best_fo     = fo_candidates[0]
    best_delays = np.zeros(nModes, dtype=int)
    best_peaks  = np.zeros(nModes)

    for fo_cand in fo_candidates:
        phase_comp  = np.exp(-1j * 2.0 * np.pi * fo_cand * n / Fs)
        delays_cand = np.zeros(nModes, dtype=int)
        peaks_cand  = np.zeros(nModes)

        for m in range(nModes):
            r_comp = rx[:, m] * phase_comp
            p      = channel_pilot_samples[:, m]

            corr = signal.correlate(r_comp, p, mode='full')
            lags = np.arange(len(corr)) - (pilotLen_samples - 1)

            # 只搜索第一帧窗口内的正延迟，避免多帧重复导频引起误对齐
            mask     = (lags >= 0) & (lags <= max_lag)
            corr_win = np.abs(corr[mask])
            lags_win = lags[mask]

            pk_idx         = int(np.argmax(corr_win))
            delays_cand[m] = int(lags_win[pk_idx])
            peaks_cand[m]  = corr_win[pk_idx]

        score = float(np.max(peaks_cand))
        if score > best_score:
            best_score  = score
            best_fo     = fo_cand
            best_delays = delays_cand.copy()
            best_peaks  = peaks_cand.copy()

    # ── Step 3：确定最终延迟，np.roll 对齐 ───────────────────────────────
    if searchMode == 'best':
        # 相关峰最高的极化模式决定公共延迟（两模共用）
        delay  = int(best_delays[int(np.argmax(best_peaks))])
        rxSync = np.roll(rx, -delay, axis=0)
    else:  # 'perMode'
        delay  = best_delays.copy()                              # (nModes,)
        rxSync = np.stack(
            [np.roll(rx[:, m], -int(delay[m])) for m in range(nModes)],
            axis=1
        )

    # ── Step 4：频偏补偿 ──────────────────────────────────────────────────
    # roll 之后 t 从 0 重置 → 引入的常数相位偏移由后级 CPE 消除
    t        = np.arange(nSamples) / Fs
    fo_comp  = np.exp(-1j * 2.0 * np.pi * best_fo * t)          # (nSamples,)
    rxSyncFO = rxSync * fo_comp[:, np.newaxis]                   # 广播到所有极化

    # ── 输出摘要 ──────────────────────────────────────────────────────────
    pilotSeq_est = pilotLen_samples // SpS
    print(f"[pilotBasedSync] FO 估计       = {best_fo / 1e6:.3f} MHz")
    print(f"[pilotBasedSync] 延迟          = {delay} 样本")
    print(f"[pilotBasedSync] 搜索窗口      = [0, {max_lag}] 样本"
          + (" (限第 1 帧)" if NSpF is not None else " (无限制)"))
    print(f"[pilotBasedSync] 导频训练区间  = rxSyncFO[0 : {pilotLen_samples}]"
          f"  ({pilotSeq_est} 个符号)")
    print(f"[pilotBasedSync] rxSyncFO.shape = {rxSyncFO.shape}")

    return rxSyncFO, delay, best_fo

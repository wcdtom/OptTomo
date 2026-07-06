"""
equalization_plus.py
====================
2×2 Fractionally-Spaced MIMO LMS Equalizer（FSE），OptiCommPy 风格参数管理。
支持 Pilot 监督训练 + Decision-Directed（DD）两阶段一体化训练。
支持可调输出采样率（SpSout）：符号输出（SpSout=1）或保持采样域输出（SpSout=SpS）。
支持 pilot 输入两种模式：symbol（符号率）或 sample（带采样率）。

输出对齐说明
------------
函数内部在输入信号**前后各填充 center = nTaps//2 个零**（中心 tap 对齐），
确保：
    y[k] 的中心 tap 恰好对准 rx[k * stride]（stride = SpS // SpSout）
    SpSout=1 时：y[k]  ↔ symbTx[k]         （符号对齐，与 Tx 严格一致）
    SpSout=SpS 时：y[k*SpS] ↔ symbTx[k]    （采样域输出，用于 CD reload）

推荐调用方式（符号输出，两阶段 DD，pilot 为符号）
--------------------------------------------------
    from optic.utils import pnorm

    rxNorm = pnorm(rxSyncFO)

    paramEq.SpS      = 2
    paramEq.SpSout   = 1          # 默认：符号输出（向后兼容）
    paramEq.nTaps    = 61
    paramEq.mu       = 1e-3
    paramEq.nTrain   = 2048
    paramEq.dd       = True
    paramEq.muDD     = 1e-4
    paramEq.nTrainDD = 10000
    paramEq.M        = 16

    y_EQ, W, err = mimoLMSFSE(rxNorm, pilotSymbols, param=paramEq)
    # pilotMode='symbol'（默认），pilotSymbols.shape = (nPilot, 2)

推荐调用方式（采样输出 + pilot 为带采样率的信号）
---------------------------------------------------
    paramEq.SpSout   = paramEq.SpS   # = 2：保持 2 采样/符号输出
    paramEq.pilotMode = 'sample'     # pilot 输入为 2 采样/符号
    paramEq.pilotSpS  = 2            # pilot 的采样率（默认 = SpS）

    y_EQ, W, err = mimoLMSFSE(rxNorm, pilotSamples, param=paramEq)
    # pilotSamples.shape = (nPilot * pilotSpS, 2)
"""

import numpy as np

try:
    from optic.utils import pnorm
except ImportError:
    def pnorm(x):
        """每列归一化到单位平均功率（fallback 实现）。"""
        power = np.mean(np.abs(x) ** 2, axis=0, keepdims=True)
        return x / np.sqrt(np.maximum(power, 1e-12))


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _center_pad(rx, nTaps):
    """
    前后各填充 center = nTaps//2 个零（中心 tap 对齐）。

    对于 nTaps=41, center=20：
        rx_pad = [0...0(20), x[0], ..., x[N-1], 0...0(20)]
        shape : (nSamples + nTaps - 1, nPolModes)

    使得对位置 k*stride 取窗 rx_pad[k*stride : k*stride+nTaps] 时，
    窗口中心 rx_pad[k*stride + center] = rx[k*stride]。
    """
    center    = nTaps // 2
    nPolModes = rx.shape[1]
    pad       = np.zeros((center, nPolModes), dtype=rx.dtype)
    return np.vstack([pad, rx, pad])


def _qamConstellation(M):
    """
    生成归一化 Square M-QAM 星座点（平均功率 = 1）。
    支持 M = 4, 16, 64, 256。
    """
    sqrtM  = int(np.round(np.sqrt(M)))
    levels = np.arange(-(sqrtM - 1), sqrtM, 2, dtype=float)
    const  = np.array([r + 1j * i for r in levels for i in levels])
    return const / np.sqrt(np.mean(np.abs(const) ** 2))


def _hardDecision(y, const):
    """
    最近邻硬判决。

    Parameters
    ----------
    y     : ndarray, shape (N, nPol)   — 待判决符号
    const : ndarray, shape (M,)        — 归一化星座点

    Returns
    -------
    decisions : ndarray, shape (N, nPol)，与 y 同尺度的星座点
    """
    nPol      = y.shape[1]
    decisions = np.zeros_like(y)
    for m in range(nPol):
        idx             = np.argmin(np.abs(y[:, m:m+1] - const[np.newaxis, :]), axis=1)
        decisions[:, m] = const[idx]
    return decisions


def _prepare_pilot(pilot, pilotMode, pilotSpS, normalize):
    """
    统一处理 pilot 输入，返回符号率参考序列。

    Parameters
    ----------
    pilot      : ndarray — pilot 输入信号
    pilotMode  : str     — 'symbol' 或 'sample'
    pilotSpS   : int     — pilot 的采样率（仅 pilotMode='sample' 时有效）
    normalize  : bool    — 是否归一化

    Returns
    -------
    d : ndarray, shape (nPilotSymbols, nPolModes)  — 符号率 pilot
    """
    if pilotMode == 'symbol':
        # pilot 已是符号率，直接使用
        d = pnorm(pilot) if normalize else pilot.copy()
    elif pilotMode == 'sample':
        # pilot 带采样率，下采样到符号率
        d = pilot[::pilotSpS, :]
        d = pnorm(d) if normalize else d
    else:
        raise ValueError(
            f"[mimoLMSFSE] pilotMode 必须为 'symbol' 或 'sample'，"
            f"当前值为 '{pilotMode}'。"
        )
    return d


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def mimoLMSFSE(rx, trainingSeq, param=None):
    """
    2×2 Fractionally-Spaced LMS (FS-LMS) MIMO 均衡器。
    支持 Pilot 监督训练 + Decision-Directed（DD）两阶段一体化训练。
    支持可调输出采样率（SpSout）。

    Parameters
    ----------
    rx : ndarray, shape (nSamples, 2)
        接收信号，分数间隔采样（SpS ≥ 2），已完成频偏补偿和归一化。
        建议传入完整 rxNorm（全帧），函数内部自动处理训练 + 推断。
    trainingSeq : ndarray
        训练参考信号，支持两种模式（由 param.pilotMode 控制）：
        - pilotMode='symbol'（默认）：shape (nPilot, 2)，符号率 pilot
          如 transmitter.pilot[:, :, ch]
        - pilotMode='sample'：shape (nPilot * pilotSpS, 2)，带采样率 pilot
          如 transmitter.pulseTraining[:, :, ch]，内部自动下采样到符号率
    param : parameters object（可选），支持以下属性：

        ── Pilot 输入模式 ─────────────────────────────────────────────────
        pilotMode : str   — 'symbol'（默认）或 'sample'
                            'symbol'：pilot 为符号率，直接用于训练
                            'sample'：pilot 带采样率，内部按 pilotSpS 下采样
        pilotSpS  : int   — pilot 的每符号采样数（仅 pilotMode='sample' 时有效）
                            默认 = SpS

        ── Pilot 阶段 ─────────────────────────────────────────────────────
        SpS       : int   — 输入每符号采样数，默认 2
        SpSout    : int   — 输出每符号采样数，默认 1（符号输出）
                            设为 SpS 则输出全采样率（用于 CD reload）
                            必须能整除 SpS（即 SpS % SpSout == 0）
        nTaps     : int   — FIR 抽头数（建议奇数），默认 21
        mu        : float — Pilot 阶段 LMS 步长，默认 1e-3
        nTrain    : int   — Pilot 监督训练符号数，默认 2000
        normalize : bool  — 是否在内部 pnorm 归一化，默认 False

        ── DD 阶段 ────────────────────────────────────────────────────────
        dd        : bool  — 是否开启 Decision-Directed 续训，默认 False
        muDD      : float — DD 阶段步长，默认 1e-4
        nTrainDD  : int   — DD 训练符号数（0 = 全部剩余），默认 0
        M         : int   — 星座阶数（DD 硬判决用），默认 16

        ── 其他 ───────────────────────────────────────────────────────────
        verbose   : bool  — 是否打印收敛信息，默认 False

    Returns
    -------
    y : ndarray
        均衡输出信号。
        SpSout=1  → shape (nSamples // SpS, 2)，符号率，y[k] ↔ symbTx[k]。
        SpSout=SpS → shape (nSamples, 2)，全采样率，y[k*SpS] ↔ symbTx[k]。
    W : ndarray, shape (2, 2, nTaps)
        最终收敛的 MIMO 滤波器系数。
    err : ndarray, shape (nTrain_actual + nTrainDD_actual, 2)
        训练误差序列（Pilot 段 + DD 段拼接）。

    Notes
    -----
    SpSout 与 stride 的关系：
        stride = SpS // SpSout
        SpSout=1  → stride=SpS → 每隔 SpS 个采样取一个输出（符号率）
        SpSout=SpS → stride=1  → 每个采样都输出（全采样率）

    训练始终在符号率进行（stride=SpS）。W 收敛后，
    在最终输出阶段以 stride=SpS//SpSout 步进对全帧做前向滤波。
    """

    # ---------- 参数读取 ----------
    SpS       = getattr(param, 'SpS',       2)
    SpSout    = getattr(param, 'SpSout',    1)
    nTaps     = getattr(param, 'nTaps',     21)
    mu        = getattr(param, 'mu',        1e-3)
    nTrain    = getattr(param, 'nTrain',    2000)
    normalize = getattr(param, 'normalize', False)
    dd        = getattr(param, 'dd',        False)
    muDD      = getattr(param, 'muDD',      1e-4)
    nTrainDD  = getattr(param, 'nTrainDD',  0)
    M         = getattr(param, 'M',         16)
    verbose   = getattr(param, 'verbose',   False)

    # ---- pilot 模式参数 ----
    pilotMode = getattr(param, 'pilotMode', 'symbol')
    pilotSpS  = getattr(param, 'pilotSpS',  2)

    # ---------- 参数校验 ----------
    if SpS % SpSout != 0:
        raise ValueError(
            f"[mimoLMSFSE] SpS={SpS} 必须能整除 SpSout={SpSout}。"
        )
    stride   = SpS // SpSout   # 输出步长（采样域）
    nSamples  = rx.shape[0]
    nPolModes = rx.shape[1]
    nSymbMax  = nSamples // SpS

    # ---------- 归一化（可选）----------
    rx = pnorm(rx) if normalize else rx

    # ---------- pilot 预处理：统一转为符号率 ----------
    d = _prepare_pilot(trainingSeq, pilotMode, pilotSpS, normalize)

    # ---------- 中心对齐零填充 ----------
    center    = nTaps // 2
    rx_pad    = _center_pad(rx, nTaps)   # shape: (nSamples + nTaps - 1, nPolModes)

    # ---------- 初始化为中心脉冲 ----------
    W = np.zeros((nPolModes, nPolModes, nTaps), dtype=complex)
    for m in range(nPolModes):
        W[m, m, center] = 1.0

    # ---------- Pilot 阶段训练符号数 ----------
    nTrain_ = min(nTrain, d.shape[0], nSymbMax)
    if nTrain_ <= 0:
        raise ValueError(
            f"Pilot 训练符号数为 0：rx.shape={rx.shape}, nTaps={nTaps}, SpS={SpS}, "
            f"nTrain={nTrain}, pilot.shape={trainingSeq.shape}。\n"
            "请检查 rx 长度（建议完整 rxNorm）和 pilot 长度。"
        )

    err_pil = np.zeros((nTrain_,  nPolModes), dtype=complex)

    # ══════════════════════════════════════════════════════════════════════
    # 阶段一：Pilot 监督训练（始终在符号率进行）
    # ══════════════════════════════════════════════════════════════════════
    for k in range(nTrain_):
        idx  = k * SpS                             # 符号 k 对应的采样位置
        x    = rx_pad[idx: idx + nTaps, :]         # (nTaps, nPolModes)
        y_k  = np.einsum('ijt,tj->i', W, x)
        e    = d[k, :] - y_k
        err_pil[k, :] = e
        W   += mu * e[:, np.newaxis, np.newaxis] * np.conj(x).T[np.newaxis, :, :]

    # ══════════════════════════════════════════════════════════════════════
    # 阶段二：Decision-Directed（可选，始终在符号率进行）
    # ══════════════════════════════════════════════════════════════════════
    err_dd = np.zeros((0, nPolModes), dtype=complex)

    if dd:
        const    = _qamConstellation(M)
        dd_start = nTrain_
        dd_end   = nSymbMax if nTrainDD == 0 else min(nTrain_ + nTrainDD, nSymbMax)
        nDD_     = dd_end - dd_start

        err_dd = np.zeros((nDD_, nPolModes), dtype=complex)

        for k in range(dd_start, dd_end):
            idx  = k * SpS
            x    = rx_pad[idx: idx + nTaps, :]
            y_k  = np.einsum('ijt,tj->i', W, x)

            d_hat = _hardDecision(y_k[np.newaxis, :], const)[0, :]
            e     = d_hat - y_k
            err_dd[k - dd_start, :] = e
            W    += muDD * e[:, np.newaxis, np.newaxis] * np.conj(x).T[np.newaxis, :, :]

    # ══════════════════════════════════════════════════════════════════════
    # 阶段三：全帧前向滤波（W 固定，输出采样率由 SpSout 决定）
    # ══════════════════════════════════════════════════════════════════════
    # stride = SpS // SpSout
    #   SpSout=1   → stride=SpS → 符号域输出（nSymbMax 个点）
    #   SpSout=SpS → stride=1   → 采样域输出（nSamples 个点）
    nOut  = nSamples // stride
    y_out = np.zeros((nOut, nPolModes), dtype=complex)

    for k in range(nOut):
        idx       = k * stride
        x         = rx_pad[idx: idx + nTaps, :]
        y_out[k]  = np.einsum('ijt,tj->i', W, x)

    # ---------- 误差拼接 ----------
    err = np.vstack([err_pil, err_dd])

    # ---------- 收敛信息 ----------
    if verbose:
        seg_p = max(1, nTrain_ // 10)
        mse_p_start = np.mean(np.abs(err_pil[:seg_p,  :]) ** 2)
        mse_p_end   = np.mean(np.abs(err_pil[-seg_p:, :]) ** 2)

        print(
            f"[mimoLMSFSE] nTaps={nTaps}, SpS={SpS}, SpSout={SpSout}, "
            f"stride={stride}, normalize={normalize}\n"
            f"  ── Pilot 模式：pilotMode='{pilotMode}'"
            + (f", pilotSpS={pilotSpS}" if pilotMode == 'sample' else "")
            + f"\n"
            f"  ── Pilot 阶段：nTrain={nTrain_}, mu={mu:.2e}\n"
            f"     MSE 初始段（前10%）: {mse_p_start:.4f}\n"
            f"     MSE 末尾段（后10%）: {mse_p_end:.4f}\n"
            f"     收敛改善: {10*np.log10(mse_p_start / max(mse_p_end, 1e-12)):.1f} dB\n"
            f"  ── 输出：{nOut} 点（SpSout={SpSout}，"
            f"{'符号率' if SpSout == 1 else f'{SpSout}采样/符号'}）"
        )

        if dd and err_dd.shape[0] > 0:
            seg_d = max(1, err_dd.shape[0] // 10)
            mse_d_start = np.mean(np.abs(err_dd[:seg_d,  :]) ** 2)
            mse_d_end   = np.mean(np.abs(err_dd[-seg_d:, :]) ** 2)
            print(
                f"  ── DD 阶段：nTrainDD={err_dd.shape[0]}, muDD={muDD:.2e}\n"
                f"     MSE 初始段（前10%）: {mse_d_start:.4f}\n"
                f"     MSE 末尾段（后10%）: {mse_d_end:.4f}\n"
                f"     收敛改善: {10*np.log10(mse_d_start / max(mse_d_end, 1e-12)):.1f} dB"
            )

    return y_out, W, err


# ---------------------------------------------------------------------------
# 辅助函数：对完整信号应用固定 W（不更新权重）
# ---------------------------------------------------------------------------

def applyMIMOFilter(synchronized_rx, W, param=None):
    """
    用训练后的 MIMO 滤波器系数对完整接收信号做线性均衡（W 固定不更新）。

    Parameters
    ----------
    synchronized_rx : ndarray, shape (nSamples, 2)
        同步后的接收信号（SpS ≥ 2）。
    W : ndarray, shape (2, 2, nTaps)
        由 mimoLMSFSE 返回的收敛滤波器系数。
    param : parameters object（可选）
        SpS       : int  — 输入每符号采样数，默认 2
        SpSout    : int  — 输出每符号采样数，默认 1（符号输出）
                          设为 SpS 则输出全采样率
        normalize : bool — 是否内部 pnorm 归一化，默认 False

    Returns
    -------
    y : ndarray
        SpSout=1  → shape (nSamples // SpS, 2)，符号率。
        SpSout=SpS → shape (nSamples, 2)，全采样率。
    """
    SpS       = getattr(param, 'SpS',       2)
    SpSout    = getattr(param, 'SpSout',    1)
    normalize = getattr(param, 'normalize', False)
    nTaps     = W.shape[2]

    if SpS % SpSout != 0:
        raise ValueError(
            f"[applyMIMOFilter] SpS={SpS} 必须能整除 SpSout={SpSout}。"
        )

    rx        = pnorm(synchronized_rx) if normalize else synchronized_rx
    nSamples  = rx.shape[0]
    nPolModes = rx.shape[1]
    stride    = SpS // SpSout
    nOut      = nSamples // stride

    rx_pad = _center_pad(rx, nTaps)

    y = np.zeros((nOut, nPolModes), dtype=complex)
    for k in range(nOut):
        idx     = k * stride
        x       = rx_pad[idx: idx + nTaps, :]
        y[k, :] = np.einsum('ijt,tj->i', W, x)

    return y

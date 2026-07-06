"""
================================================================
Pilot-aided carrier phase recovery (:mod:`carrierRecovery_plus`)
================================================================

Functions
---------
pilotCPR        -- Phase-pilot aided carrier phase recovery.

支持可调输入采样率（SpS 参数）：
    SpS=1（默认）：输入为符号率信号（向后兼容，与原版行为完全一致）
    SpS=2        ：输入为 2 采样/符号信号（用于方案B层析，CD reload 前）

当 SpS>1 时，函数在**符号域**进行相位估计（pilot 定位、星座乘积），
但相位插值和补偿作用于**全部采样点**，输出与输入等长。

"""

import numpy as np
from optic.utils import parameters


def _calPilotMask(nSym, NSpF, pilotSeq, phasePilot):
    """
    生成与 pilotWDMTx._cal_pilot_idx() 一致的 pilot mask（符号域）。

    True 表示该符号位置为已知 pilot（含帧头 pilotSeq 段和后续 phase pilot）。

    Parameters
    ----------
    nSym : int
        总符号数。
    NSpF : int
        每帧符号数（transmitter.NSpF）。
    pilotSeq : int
        帧头同步 pilot 长度。
    phasePilot : int or None
        phase pilot 插入间隔。0 或 None 表示无 phase pilot。

    Returns
    -------
    mask : ndarray of bool, shape (nSym,)
    """
    idx = np.arange(NSpF)

    if phasePilot == 0 or phasePilot is None:
        idx_pil = idx < pilotSeq
    else:
        idx_ph_pil = ((idx - pilotSeq) % phasePilot != 0) & (idx - pilotSeq > 0)
        idx_ph_pil[pilotSeq] = ~idx_ph_pil[pilotSeq]
        idx_pil = ~idx_ph_pil

    nFrames = int(np.ceil(nSym / NSpF))
    mask = np.tile(idx_pil, nFrames)[:nSym]
    return mask


def pilotCPR(y, pilotSymbols, param=None):
    """
    Pilot-aided carrier phase recovery (CPR)。

    利用已知的 pilot 符号（帧头 pilot + phase pilot）估计并补偿激光相位噪声。
    内部自动重建完整参考序列，无需调用方传入 txSymbols。

    支持两种输入采样率（由 param.SpS 控制）：
    - SpS=1（默认）：输入为符号率信号，输出为符号率。
    - SpS=2        ：输入为 2 采样/符号信号，输出保持 2 采样/符号。
                     相位估计在符号级进行（pilot 位置取中心采样），
                     插值和补偿作用于全部采样点。
                     用于方案B：FSE(SpSout=2) → CPR(SpS=2) → CD reload。

    Parameters
    ----------
    y : ndarray, shape (nTotal, nPol)
        均衡后的接收信号。
        SpS=1：nTotal = nSym（符号率）。
        SpS=2：nTotal = nSym * 2（2 采样/符号）。
    pilotSymbols : ndarray, shape (nPilotPerFrame, nPol)
        **一帧**内所有 pilot 符号（帧头 + phase pilot），
        即 ``transmitter.pilot[:, :, chIndex]``。
    param : parameters object, optional
        * **SpS** (*int*)            -- 输入每符号采样数。默认：1。
        * **NSpF** (*int*)           -- 每帧总符号数（transmitter.NSpF）。必须提供。
        * **pilotSeq** (*int*)       -- 帧头同步 pilot 长度。默认：2048。
        * **phasePilot** (*int*)     -- phase pilot 间隔。0/None 表示无。默认：0。
        * **useInitialPilot** (*bool*)-- 帧头 pilotSeq 是否纳入相位估计。默认：True。
        * **avgPols** (*bool*)       -- 两偏振相位估计是否取平均。默认：False。

    Returns
    -------
    yCPR : ndarray, shape (nTotal, nPol)
        相位补偿后的信号，与输入等长等形。
    phiInterp : ndarray, shape (nTotal, nPol)
        插值得到的相位噪声估计（弧度），与输入等长。
    pilotIdx : ndarray of int
        参与相位估计的 pilot **符号**位置索引（符号域，与 SpS 无关）。

    Notes
    -----
    SpS>1 时内部处理逻辑：
      1. 用 ``_calPilotMask`` 在符号域定位 pilot 符号（``pilotIdx_sym``）。
      2. 对应的采样位置为 ``pilotIdx_sym * SpS``（取各 pilot 符号的第 0 个采样）。
      3. 在采样位置取 y 值，与参考符号做乘积，得到 pilot 处的相位估计。
      4. 以**采样域索引**为横坐标插值到全部 nTotal 个采样点。
      5. 相位补偿作用于全部 nTotal 个采样点，输出等长信号。

    Examples
    --------
    ::

        # 方案 A（符号输出，向后兼容）
        paramCPR.SpS         = 1
        paramCPR.NSpF        = transmitter.NSpF
        paramCPR.pilotSeq    = paramTx.pilotSeq
        paramCPR.phasePilot  = paramTx.pilot_ins_rat
        yCPR, phiInterp, pilotIdx = pilotCPR(y_EQ, transmitter.pilot[:,:,ch], param=paramCPR)

        # 方案 B（2采样/符号输出，用于 CD reload 后层析）
        paramCPR.SpS         = 2          # FSE 以 SpSout=2 输出
        paramCPR.NSpF        = transmitter.NSpF
        paramCPR.pilotSeq    = paramTx.pilotSeq
        paramCPR.phasePilot  = paramTx.pilot_ins_rat
        yCPR, phiInterp, pilotIdx = pilotCPR(y_EQ2, transmitter.pilot[:,:,ch], param=paramCPR)
        # yCPR.shape = (nSamples, nPol)，可直接做 CD reload
    """
    # ── 参数读取 ──────────────────────────────────────────────
    SpS             = getattr(param, 'SpS',             1)
    NSpF            = getattr(param, 'NSpF',            None)
    pilotSeq        = getattr(param, 'pilotSeq',        2048)
    phasePilot      = getattr(param, 'phasePilot',      0)
    useInitialPilot = getattr(param, 'useInitialPilot', True)
    avgPols         = getattr(param, 'avgPols',         False)

    if NSpF is None:
        raise ValueError("[pilotCPR] 必须提供 param.NSpF（每帧符号数）。")

    # ── 维度解析 ──────────────────────────────────────────────
    nTotal    = y.shape[0]          # 总点数（采样或符号，由 SpS 决定）
    nPolModes = y.shape[1]
    nSymb     = nTotal // SpS       # 符号数（SpS=1 时 nSymb=nTotal）

    # pilotSymbols 兼容 1D（单偏振）输入
    if pilotSymbols.ndim == 1:
        pilotSymbols = pilotSymbols[:, np.newaxis]

    # ── 符号域 pilot mask ──────────────────────────────────────
    framePilotMask = _calPilotMask(NSpF, NSpF, pilotSeq, phasePilot)
    pilotMask = _calPilotMask(nSymb, NSpF, pilotSeq, phasePilot)

    if not useInitialPilot:
        framePosArr = np.arange(nSymb) % NSpF
        pilotMask   = pilotMask & (framePosArr >= pilotSeq)

    pilotIdx_sym = np.where(pilotMask)[0]   # 符号域 pilot 索引
    nPilots      = len(pilotIdx_sym)

    if nPilots < 2:
        raise ValueError(
            f"[pilotCPR] pilot 数量不足（{nPilots} 个），无法插值。"
            f" 请检查 NSpF={NSpF}, pilotSeq={pilotSeq}, phasePilot={phasePilot}。"
        )

    # ── 采样域 pilot 位置（SpS=1 时与符号域相同）──────────────
    # 取每个 pilot 符号的第 0 个采样点（idx * SpS）
    pilotIdx_domain = pilotIdx_sym * SpS    # shape (nPilots,)，采样域

    # ── 内部重建参考序列 ──────────────────────────────────────
    if not useInitialPilot:
        framePilotPos = np.where(framePilotMask)[0]
        frameActivePilotMask = framePilotMask & (np.arange(NSpF) >= pilotSeq)
        frameActivePilotPos = np.where(frameActivePilotMask)[0]
        activePilotInCompact = np.searchsorted(framePilotPos, frameActivePilotPos)
        pilotSymbolsRefFrame = pilotSymbols[activePilotInCompact, :]
    else:
        pilotSymbolsRefFrame = pilotSymbols

    nPilotPerFrame = pilotSymbolsRefFrame.shape[0]
    nFramesNeeded  = int(np.ceil(nPilots / nPilotPerFrame))
    pilotRef       = np.tile(pilotSymbolsRefFrame, (nFramesNeeded, 1))[:nPilots, :]
    # pilotRef[k, :] 对应 pilotIdx_sym[k] 处的参考符号

    # ── 相位估计插值横坐标 ──────────────────────────────────────
    # 以采样域索引为基准，插值到全部 nTotal 个点
    allIdx_domain = np.arange(nTotal)       # [0, 1, ..., nTotal-1]（采样域）

    # y 在 pilot 采样位置的值
    y_at_pilot = y[pilotIdx_domain, :]      # shape (nPilots, nPol)

    # ── 相位估计与插值 ────────────────────────────────────────
    if avgPols:
        # 两偏振合并联合估计
        zSum            = np.sum(y_at_pilot * np.conj(pilotRef), axis=1)
        phiCommon       = np.unwrap(np.angle(zSum))
        phiInterpCommon = np.interp(allIdx_domain, pilotIdx_domain, phiCommon)
        phiInterp       = np.tile(phiInterpCommon[:, np.newaxis], (1, nPolModes))
    else:
        # 每偏振独立估计（默认）
        phiInterp = np.zeros((nTotal, nPolModes))
        for m in range(nPolModes):
            z                 = y_at_pilot[:, m] * np.conj(pilotRef[:, m])
            phi               = np.unwrap(np.angle(z))
            phiInterp[:, m]   = np.interp(allIdx_domain, pilotIdx_domain, phi)

    # ── 相位补偿 ──────────────────────────────────────────────
    y_CPR = y * np.exp(-1j * phiInterp)

    # 返回符号域 pilot 索引（与 SpS 无关，便于调用方理解帧结构）
    return y_CPR, phiInterp, pilotIdx_sym

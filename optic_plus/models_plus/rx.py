import numpy as np
import scipy.constants as const
from scipy.signal import correlate
from numpy.typing import NDArray
from tqdm import tqdm
import time

from optic.dsp.core import pulseShape, firFilter, decimate, pnorm
from optic.models.devices import pdmCoherentReceiver, basicLaserModel
from optic.dsp.equalization import edc, mimoAdaptEqualizer
from optic.comm.metrics import fastBERcalc, monteCarloGMI
from optic.plot import pconst

try:
    from optic.utils import parameters
    from optic.models.channels import ssfm
except ImportError:
    def parameters():
        class _P:
            pass
        return _P()
    ssfm = None

from optic_plus.dsp_plus.synchronization_plus import pilotBasedSync
from optic_plus.dsp_plus.equalization_plus import mimoLMSFSE
from optic_plus.dsp_plus.carrierRecovery_plus import pilotCPR

class RefAidedeRP1:
    """
    RefAidedeRP1: CD-reload-capable Physical Parameter Estimation (PPE)
    via fiber tomography.

    通过光纤层析（Tomography）方法，从发送信号和接收信号估计光纤非线性系数 γ(z) 的
    逐点分布。算法核心（一阶微扰）：

        A(L) ≈ A₀(L) + G · γ
        其中 A₀ = 线性参考信号，G = 层析矩阵，γ = 待估计非线性系数

    ══════════════════════════════════════════════════════════════
    两种工作模式
    ══════════════════════════════════════════════════════════════

    afterDSP=False（默认，光学波形域）
    ─────────────────────────────────
        适用于原始光学波形（SpS > 1，光纤直接输出，未做任何 DSP）：
            A(L)  = sigRx （光纤输出）
            A₀    = D{+L}(sigTx)
            A₁    = sigRx - A₀
            G[z]  = j·Δz · D{L-z}( NL(z) )
            A(z)  = D{+z}(sigTx)

    afterDSP=True（历史参数名：CD Reload 输入模式）
    ──────────────────────────────────────────────────────────
        输入 sigRx 已经处在“色散补偿参考面”，内部自动执行 CD Reload
        （重加色散 D{+L}），还原为光学域后统一处理。

        注意：afterDSP 是兼容旧代码的参数名，不表示本函数一定执行了
        FSE/CPR 等 DSP 算法。sigRx 可以来自真实 CDC 后 DSP 链路，也可以
        来自仿真 sanity check 中的 D{-L}(SSFM output)。后者只能称为
        CD-reload/oracle 验证，不能称为 after-DSP 验证。

        ★ 内部自动执行整数采样对齐，并默认只报告常数相位，不自动移除，
          确保 sigTx 和 sigRx 时间同步后再做 CD Reload。

            sigRx_input = 色散已补偿的接收场（SpS 可为 2 或更高）
            A[L]  = D{+L}(sigRx_input)    ← CD Reload（内部自动完成）
            A₀    = D{+L}(sigTx)
            A₁    = A[L] - A₀
            G[z]  = j·Δz · D{L-z}( NL(z) )  ← 与光学域完全一致

        论文稳定条件：|β₂|·BW²·Δz > 0.078
        → SpS=2, Rs=32GBd, Δz≥2km 可满足。

    ══════════════════════════════════════════════════════════════
    Manakov 非线性算符（双偏振自动适配）
    ══════════════════════════════════════════════════════════════

    nModes ≥ 2 时自动使用 Manakov 非线性算符：
        NL_m(z) = (8/9) · (|Ax(z)|² + |Ay(z)|²) · A_m(z)

    nModes = 1 时使用标量 NLSE：
        NL(z) = |A(z)|² · A(z)

    8/9 因子与 OptiCommPy 的 manakovSSF 保持一致。

    ══════════════════════════════════════════════════════════════
    完整调用示例（CD Reload 输入模式）
    ══════════════════════════════════════════════════════════════

        # ── FSE + CPR 输出 SpS=2 ──
        paramEq.SpS = 2;  paramEq.SpSout = 2;  paramEq.dd = False
        y_EQ2, _, _ = mimoLMSFSE(rxNorm, pilotSamples, param=paramEq)

        paramCPR.SpS = 2
        yCPR2, _, _ = pilotCPR(y_EQ2, transmitter.pilot[:,:,ch], param=paramCPR)
        #                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                      ★ 必须传入紧凑 pilot 数组，不是 resampledPilot！

        # ── 层析参数 ──
        param           = parameters()
        param.D         = 16              # ps/nm/km
        param.alpha     = 0.20            # dB/km
        param.Fc        = 193.1e12        # Hz
        param.gamma     = 1.3             # 1/W/km
        param.Fs        = 2 * paramTx.Rs  # SpS=2 的采样率
        param.Ltotal    = paramCh.Ltotal  # ← 必须传入！
        param.Lspan     = paramCh.Lspan
        param.deltaZ    = 2.0             # SpS=2 时需要 ≥ 2km
        param.afterDSP  = True            # ← 历史参数名：启用 CD Reload

        paramRun = parameters()
        paramRun.lambdaReg = 0
        paramRun.normalize = False
        paramRun.plot = True
        paramRun.allModes = True
        paramRun.referenceMatched = True

        ppe = RefAidedeRP1(param=param)
        gammaEst, G = ppe.run(sigTx_2sps, yCPR2, param=paramRun)

    Parameters
    ----------
    param : parameters object（可选），支持以下属性：

        ── 光纤物理参数 ────────────────────────────────────────────
        D        : float  色散系数 [ps/nm/km]，默认 16
        alpha    : float  损耗系数 [dB/km]，默认 0.20
        Fc       : float  载波频率 [Hz]，默认 193.1e12
        gamma    : float  非线性系数（对比参考值）[1/W/km]，默认 1.3

        ── 仿真参数 ────────────────────────────────────────────────
        Fs       : float  采样率 [Hz]（必填）
        Ltotal   : float  链路总长 [km]（必填，无默认值）
        Lspan    : float  跨段长度 [km]，默认 50
        deltaZ   : float  层析步长 [km]，默认 0.5
        Rs       : float  符号率 [baud]，默认 32e9（用于 EDC/CD Reload 滤波器长度）
        afterDSP : bool   CD Reload 输入模式，默认 False

        ── SSFM 参数（propagate 方法使用）──────────────────────────
        hz       : float  SSFM 步长 [km]，默认 0.05
        amp      : str    放大器类型，默认 'ideal'
        prgsBar  : bool   是否显示进度条，默认 True
    """

    def __init__(self, param=None):
        if param is None:
            param = parameters()

        # ── 光纤物理参数 ─────────────────────────────────────────
        self.D     = getattr(param, 'D',     16)
        self.alpha = getattr(param, 'alpha', 0.20)
        self.Fc    = getattr(param, 'Fc',    193.1e12)
        self.gamma = getattr(param, 'gamma', 1.3)

        # ── 仿真参数（Fs 和 Ltotal 为必填）─────────────────────────
        self.Fs = getattr(param, 'Fs', None)
        if self.Fs is None:
            raise ValueError("[RefAidedeRP1] param.Fs（采样率）为必填参数！")

        self.Ltotal = getattr(param, 'Ltotal', None)
        if self.Ltotal is None:
            raise ValueError("[RefAidedeRP1] param.Ltotal（链路总长）为必填参数！")

        self.Lspan    = getattr(param, 'Lspan',    50)
        self.deltaZ   = getattr(param, 'deltaZ',   0.5)
        self.afterDSP = getattr(param, 'afterDSP', False)
        self.Rs       = getattr(param, 'Rs',       32e9)

        # ── SSFM 参数 ────────────────────────────────────────────
        self.hz      = getattr(param, 'hz',      0.05)
        self.amp     = getattr(param, 'amp',     'ideal')
        self.prgsBar = getattr(param, 'prgsBar', True)

        self._param = param

        # ── 派生常数 ─────────────────────────────────────────────
        c_kms          = const.c / 1e3
        wavelength     = c_kms / self.Fc          # 波长 [km]
        self.beta2     = -(self.D * wavelength ** 2) / (2 * np.pi * c_kms)
        self.alphaTomo = self.alpha / (10 * np.log10(np.exp(1)))   # Np/km

        # ── 层析 z 坐标序列 ──────────────────────────────────────
        self.zTomoBank = np.arange(0, self.Ltotal, self.deltaZ)    # (nZ,)

        # ── 功率归一化状态（run 时设置）──────────────────────────
        self._pAvgTx = None
        self._pAvgRx = None
        self._normalizedRun = False
        self._lastAugmentedX = None
        self._lastAugmentedCFactor = None
        self._lastRegressionInfo = None
        self._lastRunSigTx = None
        self._lastRunSigRx = None
        self._lastRunPowerTx = 1.0
        self._lastRunPowerRx = 1.0

    # ----------------------------------------------------------------
    # 内部辅助
    # ----------------------------------------------------------------

    def _to2d(self, sig: NDArray):
        """统一转为 (N, nModes)，返回 (sig2d, squeezed)。"""
        if sig.ndim == 1:
            return sig.reshape(-1, 1), True
        return sig, False

    # ----------------------------------------------------------------
    # CD 传播 / CD Reload
    # ----------------------------------------------------------------

    def _edcCD(self, sigInput: NDArray, length: float, Fs: float = None) -> NDArray:
        """
        OptiCommPy EDC 风格的色散传播。

        edc(param.L>0) 表示 CD compensation，因此这里用 L=-length
        表示正向重加色散 D{+length}。
        """
        if Fs is None:
            Fs = self.Fs
        if length == 0:
            return sigInput.copy()

        paramEDC = parameters()
        paramEDC.L  = -length
        paramEDC.D  = self.D
        paramEDC.Fc = self.Fc
        paramEDC.Fs = Fs
        paramEDC.Rs = self.Rs
        filterMemory = 6.67 * np.abs(self.beta2) * np.abs(length) * self.Rs ** 2
        paramEDC.NfilterCoeffs = int(
            2 * np.ceil(filterMemory * (Fs / self.Rs))
        )
        return edc(sigInput, paramEDC)

    def _linearCD(self, sigInput: NDArray, length: float, Fs: float = None) -> NDArray:
        """OptiCommPy EDC 风格的层析线性色散算子。"""
        return self._edcCD(sigInput, length=length, Fs=Fs)

    def cdReload(self, sigInput: NDArray, Ltotal: float = None, Fs: float = None) -> NDArray:
        """
        CD Reload：对色散已补偿的输入信号重加色散 D{+L}，
        得到等效光纤末端波形 A[L]。

        Parameters
        ----------
        sigInput : ndarray  色散已补偿参考面上的信号
        Ltotal   : float    重加色散距离 [km]；None 时使用 self.Ltotal
        Fs       : float    信号采样率 [Hz]；None 时使用 self.Fs

        Returns
        -------
        sigReloaded : ndarray  等效光纤末端波形 A[L]
        """
        if Ltotal is None:
            Ltotal = self.Ltotal
        if Fs is None:
            Fs = self.Fs

        return self._edcCD(sigInput, length=Ltotal, Fs=Fs)

    # ----------------------------------------------------------------
    # 信号自动对齐（CD Reload 输入模式专用）
    # ----------------------------------------------------------------

    def _alignSignals(self, sigTx: NDArray, sigRx: NDArray,
                      maxShift: int = 200,
                      phaseAlign: bool = False,
                      verbose: bool = True) -> tuple:
        """
        自动对齐 sigTx 和 sigRx（afterDSP=True/CD Reload 输入模式时使用）。

        接收链路或仿真 CD 补偿链路可能引入少量采样偏移和恒定相位旋转。
        即使只差 1 个采样，经过 CD Reload 后残差 A₁ 会被完全主导。
        但常数相位可能包含非线性相位的一阶成分，因此默认只报告相位，
        不自动旋转 Rx。若确认该相位来自 LO/CPR 残差，可显式启用 phaseAlign。

        步骤：
        1. FFT 互相关：找到整数采样延迟
        2. 裁剪对齐：移除延迟偏移
        3. 可选恒定相位对齐：最小化 ||sigRx - e^{jθ}·sigTx||²

        Parameters
        ----------
        sigTx    : ndarray, shape (N1, nModes)
        sigRx    : ndarray, shape (N2, nModes)
        maxShift : int    最大搜索范围 [采样数]
        phaseAlign : bool 是否把 sigRx 旋转到 sigTx 的常数相位，默认 False
        verbose  : bool   是否打印对齐信息

        Returns
        -------
        sigTx_al, sigRx_al : 对齐后的信号（等长；仅 phaseAlign=True 时相位一致）
        """
        nModes = sigTx.shape[1]
        N = min(sigTx.shape[0], sigRx.shape[0])
        nFFT = 2 * N

        # ── FFT 互相关（逐模取绝对值求和，处理偏振间相位差）──
        cc = np.zeros(nFFT)
        for m in range(nModes):
            Ftx = np.fft.fft(sigTx[:N, m], n=nFFT)
            Frx = np.fft.fft(sigRx[:N, m], n=nFFT)
            cc += np.abs(np.fft.ifft(np.conj(Frx) * Ftx))

        # ── 限制搜索范围 ──
        mask = np.zeros(nFFT, dtype=bool)
        mask[:maxShift + 1] = True            # 正延迟 0..maxShift
        mask[nFFT - maxShift:] = True         # 负延迟 -maxShift..-1
        cc[~mask] = 0

        bestIdx = np.argmax(cc)
        delay = bestIdx if bestIdx <= maxShift else bestIdx - nFFT

        # ── 裁剪对齐 ──
        if delay > 0:
            # sigTx 在 sigRx 前面 delay 个采样
            sigTx_al = sigTx[delay:]
            sigRx_al = sigRx[:sigTx_al.shape[0]]
        elif delay < 0:
            # sigRx 在 sigTx 前面 |delay| 个采样
            sigRx_al = sigRx[-delay:]
            sigTx_al = sigTx[:sigRx_al.shape[0]]
        else:
            sigTx_al = sigTx[:N]
            sigRx_al = sigRx[:N]

        nAlign = min(sigTx_al.shape[0], sigRx_al.shape[0])
        sigTx_al = sigTx_al[:nAlign]
        sigRx_al = sigRx_al[:nAlign]

        # ── 恒定相位估计（默认不校正）──────────────────────────────
        # 找到 θ 使 ||sigRx·e^{-jθ} - sigTx||² 最小
        # 最优解 θ = -angle(conj(sigRx)·sigTx)，校正为 sigRx·e^{+j·angle(cross)}
        cross_modes = np.sum(np.conj(sigRx_al) * sigTx_al, axis=0)
        cross = np.sum(cross_modes)
        theta = np.angle(cross)
        if phaseAlign:
            sigRx_al = sigRx_al * np.exp(+1j * theta)

        # ── 对齐质量指标 ──
        E_rx = np.sum(np.abs(sigRx_al) ** 2)
        E_tx = np.sum(np.abs(sigTx_al) ** 2)
        corrCoeff = float(np.sum(np.abs(cross_modes)) /
                          np.sqrt(E_rx * E_tx + 1e-30))

        residualPower = float(np.mean(np.abs(sigRx_al - sigTx_al) ** 2))
        signalPower   = float(np.mean(np.abs(sigTx_al) ** 2))
        snrEstimate   = 10 * np.log10(signalPower / max(residualPower, 1e-30))

        if verbose:
            print(f"[RefAidedeRP1] 自动对齐: "
                  f"delay={delay} 采样, "
                  f"phase={np.degrees(theta):.1f}° "
                  f"({'applied' if phaseAlign else 'reported only'}), "
                  f"corr={corrCoeff:.4f}, "
                  f"SNR_est={snrEstimate:.1f} dB")

            if corrCoeff < 0.7:
                print(f"  ⚠️ 相关系数过低 ({corrCoeff:.4f} < 0.7)！可能原因：")
                print(f"     - 接收信号尚未完成同步/频偏校正")
                print(f"     - 相干接收前端复增益尚未标定到 Tx reference")
                print(f"     - 信号长度或帧起点不匹配（sigTx 和 sigRx 来自不同区段）")

        return sigTx_al, sigRx_al

    # ----------------------------------------------------------------
    # Tomography waveform regression
    # ----------------------------------------------------------------

    def prepareTomographyWaveform(self,
                                  sigRx: NDArray,
                                  sigTx: NDArray,
                                  param=None,
                                  verbose: bool = True) -> tuple:
        """
        Tomography 前的 2-sps 物理波形回归。

        该步骤用于 tomography 前的参考面整理：
            1. Rx/Tx 分别 pnorm
            2. 每个偏振独立用互相关修复整数采样偏移
            3. 截取一段 16QAM 数据段，而不是用 pilotSeq 作为训练长度
            4. mimoAdaptEqualizer(SpS=1, alg=['nlms']) 逐样本回归到 Tx 2-sps
            5. 每个偏振消除残余常数相位并 pnorm

        Parameters
        ----------
        sigRx, sigTx : ndarray
            已经 EDC/sync 后的 Rx 2-sps 波形，以及对应的 Tx 2-sps 参考波形。
        param : parameters object
            regressionTrainSymbols : int, 16QAM 数据符号长度。默认使用可用长度。
            regressionStartSymbol  : int, 数据段起点（符号）。默认 0。
            regressionSpS          : int, 每符号采样数。默认 round(self.Fs/self.Rs)。
            regressionNTaps        : int, MIMO taps。默认 15。
            regressionNumIter      : int, 预收敛迭代次数。默认 2。
            regressionMu           : float, NLMS 步长。默认 1e-3。
            regressionM            : int, QAM 阶数。默认 16。
            regressionNormalize    : bool, 是否对输出再次 pnorm。默认 True。

        Returns
        -------
        sigTxData, sigRxRegression, info
            均为截取后的数据段长度。info 会同时保存到 self._lastRegressionInfo。
        """
        if param is None:
            param = parameters()

        rx = pnorm(np.asarray(sigRx))
        tx = pnorm(np.asarray(sigTx))
        rx, _ = self._to2d(rx)
        tx, _ = self._to2d(tx)

        nMin = min(rx.shape[0], tx.shape[0])
        rx = rx[:nMin, :]
        tx = tx[:nMin, :]

        rxRepaired = np.zeros_like(rx)
        delays = []
        for m in range(rx.shape[1]):
            corr = np.abs(correlate(rx[:, m], tx[:, m], mode='full'))
            delay = int(np.argmax(corr) - len(tx[:, m]) + 1)
            delays.append(delay)
            rxRepaired[:, m] = np.roll(rx[:, m], -delay)

        regressionSpS = int(getattr(param, 'regressionSpS',
                                    max(1, int(round(self.Fs / self.Rs)))))
        startSymbol = int(getattr(param, 'regressionStartSymbol', 0))
        trainSymbols = getattr(param, 'regressionTrainSymbols', None)
        startSample = int(getattr(param, 'regressionStartSample',
                                  startSymbol * regressionSpS))
        if trainSymbols is None:
            trainSamples = int(getattr(param, 'regressionTrainSamples',
                                       nMin - startSample))
        else:
            trainSamples = int(getattr(param, 'regressionTrainSamples',
                                       int(trainSymbols) * regressionSpS))

        dataMask = getattr(param, 'regressionDataMask', None)
        if dataMask is not None:
            dataMask = np.asarray(dataMask, dtype=bool).reshape(-1)
            nSymbolsAvailable = min(len(dataMask), nMin // regressionSpS)
            dataMask = dataMask[:nSymbolsAvailable]
            minRun = int(np.ceil(trainSamples / regressionSpS))
            startSearch = max(0, min(startSymbol, nSymbolsAvailable - 1))
            bestStart = None
            bestLen = 0
            runStart = None
            runLen = 0
            for idxSym in range(startSearch, nSymbolsAvailable):
                if dataMask[idxSym]:
                    if runStart is None:
                        runStart = idxSym
                        runLen = 0
                    runLen += 1
                    if runLen > bestLen:
                        bestStart = runStart
                        bestLen = runLen
                    if runLen >= minRun:
                        bestStart = runStart
                        bestLen = runLen
                        break
                else:
                    runStart = None
                    runLen = 0
            if bestStart is None:
                raise ValueError("[RefAidedeRP1] regression mask 中找不到连续 16QAM 数据段。")
            if bestLen < minRun and bool(getattr(param, 'regressionRequireFullDataRun', True)):
                raise ValueError(
                    "[RefAidedeRP1] regression mask 中最长连续 16QAM 数据段不足："
                    f"需要 {minRun} symbols，找到 {bestLen} symbols。"
                )
            startSample = bestStart * regressionSpS
            trainSamples = min(trainSamples, bestLen * regressionSpS)

        startSample = max(0, min(startSample, nMin - 1))
        stopSample = min(nMin, startSample + trainSamples)
        if stopSample <= startSample:
            raise ValueError("[RefAidedeRP1] tomography regression 数据段为空，请检查 "
                             "regressionStartSymbol/regressionTrainSymbols。")

        rxData = rxRepaired[startSample:stopSample, :]
        txData = tx[startSample:stopSample, :]

        paramEq = parameters()
        paramEq.SpS = 1
        paramEq.nTaps = int(getattr(param, 'regressionNTaps', 15))
        paramEq.numIter = int(getattr(param, 'regressionNumIter', 2))
        paramEq.storeCoeff = bool(getattr(param, 'regressionStoreCoeff', False))
        paramEq.M = int(getattr(param, 'regressionM', 16))
        paramEq.shapingFactor = getattr(param, 'regressionShapingFactor', 0)
        paramEq.prgsBar = bool(getattr(param, 'regressionPrgsBar', False))
        paramEq.alg = getattr(param, 'regressionAlg', ['nlms'])
        paramEq.mu = [float(getattr(param, 'regressionMu', 1e-3))]
        paramEq.L = [len(rxData)]
        paramEq.returnResults = True

        yEq, H, errSq, Hiter = mimoAdaptEqualizer(rxData, paramEq, dx=txData)

        yRegression = np.zeros_like(yEq)
        phaseDiffs = []
        for m in range(yEq.shape[1]):
            phaseDiff = np.mean(yEq[:, m] * np.conj(txData[:, m]))
            phaseDiffs.append(np.angle(phaseDiff))
            yRegression[:, m] = yEq[:, m] * np.exp(-1j * np.angle(phaseDiff))

        if bool(getattr(param, 'regressionNormalize', True)):
            yRegression = pnorm(yRegression)

        tail = min(1000, errSq.shape[1])
        info = {
            'regressionDelays': np.array(delays),
            'regressionPhase': np.array(phaseDiffs),
            'regressionStartSample': startSample,
            'regressionStopSample': stopSample,
            'regressionStartSymbol': startSample // regressionSpS,
            'regressionTrainSamples': stopSample - startSample,
            'regressionTrainSymbols': (stopSample - startSample) // regressionSpS,
            'regressionMse': float(np.nanmean(errSq).real),
            'regressionMseTail': float(np.nanmean(errSq[:, -tail:]).real),
            'regressionH': H,
            'regressionHiter': Hiter,
        }
        self._lastRegressionInfo = info

        if verbose:
            print("[RefAidedeRP1] tomography waveform regression:")
            print(f"  data segment samples = [{startSample}:{stopSample}] "
                  f"({info['regressionTrainSymbols']} symbols @ {regressionSpS} sps)")
            print(f"  delays = {delays}, "
                  f"phase = {np.degrees(phaseDiffs)} deg")
            print(f"  MSE = {info['regressionMse']:.4e}, "
                  f"tail MSE = {info['regressionMseTail']:.4e}")

        return txData, yRegression, info

    # ----------------------------------------------------------------
    # G 矩阵构建
    # ----------------------------------------------------------------

    def generateMatrixG(self, sigTx: NDArray, verbose: bool = True) -> NDArray:
        """
        构建层析矩阵 G（支持多模 + Manakov 8/9）。

        无论输入是否需要 CD Reload，G 矩阵统一使用光学域公式：
            g_m(z) = j·Δz · D{L-z}( NL_m(z) )
            A(z) = D{+z}( sigTx )

        非线性算符（自动适配）：
            nModes=1 ：NL(z) = |A(z)|² · A(z)          ← 标量 NLSE
            nModes≥2 ：NL_m(z) = (8/9)·(Σ|A_k|²)·A_m  ← Manakov

        Parameters
        ----------
        sigTx   : ndarray, shape (N,) 或 (N, nModes)
        verbose : bool  是否显示进度条，默认 True

        Returns
        -------
        G : ndarray, shape (nModes, nZ, N)
        """
        sig2d, _ = self._to2d(sigTx)
        nSamples, nModes = sig2d.shape
        nZ = len(self.zTomoBank)

        G = np.zeros((nModes, nZ, nSamples), dtype=complex)

        # Manakov 因子：双偏振 → 8/9，单偏振 → 1
        manakovFactor = 8.0 / 9.0 if nModes >= 2 else 1.0

        if verbose:
            mode_desc = f"Manakov (8/9={manakovFactor:.4f})" if nModes >= 2 else "scalar NLSE"
            print(f"[RefAidedeRP1] Building G: nZ={nZ}, "
                  f"nSamples={nSamples}, nModes={nModes}, {mode_desc}")

        iterator = tqdm(enumerate(self.zTomoBank), total=nZ,
                        desc='[RefAidedeRP1] Building G') if verbose else \
                   enumerate(self.zTomoBank)

        for zIdx, zTomo in iterator:
            # A(z)：信号正向传播到 z 处
            sigAtZ = self._linearCD(sig2d, length=zTomo)          # (N, nModes)

            # 非线性算符
            if nModes >= 2:
                # Manakov: (8/9)·(|Ax|²+|Ay|²)·A_m
                totalPower = np.sum(np.abs(sigAtZ) ** 2, axis=1, keepdims=True)  # (N, 1)
                totalPower = totalPower - 1.5 * np.mean(totalPower)
                nonlinearOp = manakovFactor * totalPower * sigAtZ                # (N, nModes)
            else:
                # 标量 NLSE: |A|²·A
                power = np.abs(sigAtZ) ** 2
                power = power - 1.5 * np.mean(power)
                nonlinearOp = power * sigAtZ                                     # (N, 1)

            # D{L-z}（正向传播剩余距离到光纤末端）
            result = self._linearCD(nonlinearOp, length=self.Ltotal - zTomo) * (1j * self.deltaZ)

            G[:, zIdx, :] = result.T                               # (nModes, N)

        return G    # (nModes, nZ, N)

    # ----------------------------------------------------------------
    # 求解 γ(z)
    # ----------------------------------------------------------------

    def solveGamma(self,
                   matrixG:   NDArray,
                   sigTx:     NDArray,
                   sigRx:     NDArray,
                   lambdaReg  = 0,
                   verbose:   bool  = True) -> NDArray:
        """
        联合多模正则化最小二乘求解 γ(z)。

            γ = Re[Σ_m Gₘᴴ Gₘ + λI]⁻¹ · Re[Σ_m Gₘᴴ A₁_m]

        注意：sigTx/sigRx 应已经过 _alignSignals 对齐（CD Reload 输入模式时）。

        Parameters
        ----------
        matrixG   : ndarray, shape (nModes, nZ, N)
        sigTx     : ndarray, shape (N,) 或 (N, nModes)  — 已对齐的 Tx 参考
        sigRx     : ndarray, shape (N,) 或 (N, nModes)  — 已对齐的 Rx 信号
        lambdaReg : float 或 'auto'
                    Tikhonov 正则化系数。默认 0。
                    'auto' → 基于 G†G 对角线和噪声估计自动选择
        verbose   : bool   是否打印诊断信息，默认 True

        Returns
        -------
        gammaEst : ndarray, shape (nZ,)
        """
        sig2dTx, _ = self._to2d(sigTx)
        sig2dRx, _ = self._to2d(sigRx)
        nModes = sig2dTx.shape[1]

        # A₀ = D{+L}(sigTx)
        A0 = self._linearCD(sig2dTx, length=self.Ltotal)

        if self.afterDSP:
            # CD Reload
            AL = self.cdReload(sig2dRx)
        else:
            AL = sig2dRx

        nZ = matrixG.shape[1]
        Gcols = [
            matrixG[:, zIdx, :].T.reshape(-1)
            for zIdx in range(nZ)
        ]
        H = np.column_stack(Gcols + [A0.reshape(-1)])
        target = AL.reshape(-1)

        HdagH = np.dot(np.conj(H).T, H)
        maxDiag = float(np.max(np.abs(np.diag(HdagH))))
        if maxDiag == 0:
            maxDiag = 1.0
        if lambdaReg != 0:
            HdagH += lambdaReg * maxDiag * np.eye(HdagH.shape[0])

        HdagA = np.dot(np.conj(H).T, target)
        xVec = np.linalg.solve(HdagH, HdagA)
        cFactor = xVec[-1]
        self._lastAugmentedX = xVec
        self._lastAugmentedCFactor = cFactor
        gammaEst = np.real(xVec[:-1] / cFactor)

        if verbose:
            recon = (H[:, :-1] @ xVec[:-1] + cFactor * A0.reshape(-1)).reshape(A0.shape)
            relErr = np.linalg.norm(AL - recon) / max(np.linalg.norm(AL), 1e-30)
            print("[RefAidedeRP1] augmented solve [G,A0]:")
            print(f"  cFactor = {cFactor.real:+.4e}{cFactor.imag:+.4e}j")
            print(f"  relative waveform fit error = {relErr:.4e}")
            print(f"  cond(H†H) = {np.linalg.cond(HdagH):.2e}")

        return gammaEst

    def _diagnoseAfterDSPReference(self,
                                   sigTx: NDArray,
                                   sigRx: NDArray,
                                   referenceMatched: bool = False,
                                   strictReference: bool = False,
                                   verbose: bool = True) -> None:
        """
        afterDSP=True（CD Reload 输入模式）时的参考面检查。

        这里的 sigRx 应该是“已去色散后的光场”，sigTx 应该是与它处在同一
        线性参考面上的 Tx 参考。如果 Rx 经过了 FSE、MIMO 解混、相位
        插值等处理，而 Tx 没有经过同一个线性算子，则 A1=CDReload(Rx)-D(L)Tx
        不再只代表光纤非线性。
        """
        if not verbose and not strictReference:
            return

        cross_modes = np.sum(np.conj(sigRx) * sigTx, axis=0)
        cross = np.sum(cross_modes)
        e_rx = float(np.sum(np.abs(sigRx) ** 2))
        e_tx = float(np.sum(np.abs(sigTx) ** 2))
        corr = float(np.sum(np.abs(cross_modes)) /
                     np.sqrt(max(e_rx * e_tx, 1e-30)))

        phase = np.angle(cross)
        sigRx_phase = sigRx * np.exp(-1j * phase)
        mismatch = float(
            np.linalg.norm(sigRx_phase - sigTx) /
            max(np.linalg.norm(sigTx), 1e-30)
        )

        if verbose:
            print("[RefAidedeRP1] CD-reload 输入参考面检查:")
            print(f"  Tx/Rx compensated-domain corr = {corr:.4f}")
            print(f"  Tx/Rx compensated-domain mismatch = {mismatch:.4f}")
            if referenceMatched:
                print("  调用方声明: sigTx 已与 sigRx 处在同一线性参考面。")
            else:
                print("  注意: 未声明 sigTx 已经过与 sigRx 相同的线性处理。")
                print("        建议先完成同步/频偏校正，并只在确定性的线性接收前端后验证。")
                print("        FSE/CPR 通常会改变参考面；CPR 不是可直接作用到 Tx 的线性算子。")

        bad_reference = (not referenceMatched) or corr < 0.7 or mismatch > 0.8
        if strictReference and bad_reference:
            raise ValueError(
                "[RefAidedeRP1] CD-reload 输入参考面检查失败：sigTx/sigRx "
                "看起来不在同一参考面。请传入 matched Tx reference，或设置 "
                "referenceMatched=True 明确承担该假设。"
            )

    # ----------------------------------------------------------------
    # 一步运行
    # ----------------------------------------------------------------

    def run(self,
            sigTx,
            sigRx: NDArray = None,
            param=None,
            **kwargs) -> tuple:
        """
        一步完成层析估计：（归一化 → 对齐 →）generateMatrixG → solveGamma → （可选）绘图。

        OptiCommPy 风格调用：

            paramRun = parameters()
            paramRun.lambdaReg = 0
            paramRun.normalize = False
            paramRun.plot = True
            gammaEst, G = ppe.run(sigTx, sigRx, param=paramRun)

        Parameters
        ----------
        sigTx      : ndarray, shape (N,) 或 (N, nModes)
                     发送信号（符号或采样，需与 sigRx 相同采样率）
        sigRx      : ndarray, shape (N,) 或 (N, nModes)
                     afterDSP=True 时为色散已补偿输入（内部自动 CD Reload + 对齐）
                     afterDSP=False 时为光纤直接输出波形
        param      : parameters object（可选），支持以下属性：
            lambdaReg        : float 或 'auto'，Tikhonov 正则化系数，默认 0
            normalize        : bool，是否分别归一化 Tx/Rx，默认 False
            verbose          : bool，是否显示进度条及诊断信息，默认 True
            plot             : bool，是否绘图，默认 False
            pAvgTx           : float，绘图归一化用 Tx 平均功率
            interval         : ndarray，绘图时域样点索引
            allModes         : bool，True=绘制全部偏振模，默认 True
            referenceMatched : bool，afterDSP=True 时声明 Tx/Rx 处在同一参考面
            strictReference  : bool，参考面检查失败时抛出 ValueError
            phaseAlign       : bool，CD Reload 输入模式时是否对 Rx 做常数相位旋转，默认 False
            FsTx, FsRx       : float，可选。若提供，将检查 Tx/Rx 采样率是否一致，
                               且是否等于 self.Fs
            waveformRegression : bool，True 时先执行 tomography waveform regression。
            regressionTrainSymbols/regressionStartSymbol/regressionSpS
                             : waveform regression 数据段和均衡器参数，
                               见 prepareTomographyWaveform。
            也可直接传入 pilotWDMRx 的单通道结果字典作为 sigTx，此时
            run 会自动使用 resampledSigTxMatched/rxSyncFO，并在需要时
            从 dataSymbolMask 填充 regressionDataMask。

        Returns
        -------
        gammaEst : ndarray, shape (nZ,)
        G        : ndarray, shape (nModes, nZ, N)
        """
        if param is None:
            param = parameters()

        # 兼容旧调用：ppe.run(sigTx, sigRx, lambdaReg=..., plot=...)
        # 新代码建议统一把这些字段放进 param。
        if kwargs:
            for key, value in kwargs.items():
                setattr(param, key, value)

        lambdaReg = getattr(param, 'lambdaReg', 0)
        normalize = getattr(param, 'normalize', False)
        verbose = getattr(param, 'verbose', True)
        plot = getattr(param, 'plot', False)
        pAvgTx = getattr(param, 'pAvgTx', None)
        interval = getattr(param, 'interval', None)
        allModes = getattr(param, 'allModes', True)
        referenceMatched = getattr(param, 'referenceMatched', False)
        strictReference = getattr(param, 'strictReference', False)
        phaseAlign = getattr(param, 'phaseAlign', False)
        fsTx = getattr(param, 'FsTx', None)
        fsRx = getattr(param, 'FsRx', None)
        waveformRegression = getattr(param, 'waveformRegression', False)

        dspResult = None
        if isinstance(sigTx, dict):
            if sigRx is not None:
                raise ValueError("[RefAidedeRP1] 传入 pilotWDMRx 结果字典时，"
                                 "sigRx 应留空。")
            dspResult = sigTx
            sigTx = np.asarray(dspResult.get("resampledSigTxMatched",
                                             dspResult["resampledSigTx"]))
            sigRx = np.asarray(dspResult["rxSyncFO"])
            if (waveformRegression
                    and not hasattr(param, 'regressionDataMask')
                    and "dataSymbolMask" in dspResult):
                param.regressionDataMask = dspResult["dataSymbolMask"]

        sig2dTx, _ = self._to2d(sigTx)
        sig2dRx, _ = self._to2d(sigRx)

        if waveformRegression:
            sig2dTx, sig2dRx, regressionInfo = self.prepareTomographyWaveform(
                sig2dRx, sig2dTx, param=param, verbose=verbose
            )
            if dspResult is not None:
                dspResult.update(regressionInfo)

        if verbose:
            print(f"[RefAidedeRP1] tomography options: "
                  f"linearCD=edc, pbarCoeff=1.5, solve=augmented")

        # ── 采样率一致性检查（数组本身无法携带 Fs，只能检查调用方声明）──
        if fsTx is not None or fsRx is not None:
            if fsTx is None or fsRx is None:
                raise ValueError("[RefAidedeRP1] 若提供采样率检查，"
                                 "param.FsTx 和 param.FsRx 必须同时提供。")
            if not np.isclose(float(fsTx), float(fsRx), rtol=1e-12, atol=0.0):
                raise ValueError("[RefAidedeRP1] sigTx/sigRx 采样率不一致: "
                                 f"FsTx={fsTx:.6e}, FsRx={fsRx:.6e}")
            if not np.isclose(float(fsTx), float(self.Fs), rtol=1e-12, atol=0.0):
                raise ValueError("[RefAidedeRP1] 输入采样率与 param.Fs 不一致: "
                                 f"FsTx=FsRx={fsTx:.6e}, param.Fs={self.Fs:.6e}")
            if verbose:
                print(f"[RefAidedeRP1] 采样率检查: "
                      f"FsTx=FsRx=param.Fs={self.Fs:.6e} Hz")

        # ── 长度对齐（粗裁剪）──────────────────────────────────────
        nMin = min(sig2dTx.shape[0], sig2dRx.shape[0])
        if sig2dTx.shape[0] != sig2dRx.shape[0]:
            if verbose:
                print(f"[RefAidedeRP1] 警告: sigTx({sig2dTx.shape[0]}) "
                      f"与 sigRx({sig2dRx.shape[0]}) 长度不同，截取到 {nMin}")
            sig2dTx = sig2dTx[:nMin]
            sig2dRx = sig2dRx[:nMin]

        # ── 功率归一化 ────────────────────────────────────────────
        self._normalizedRun = bool(normalize)
        if normalize:
            if verbose:
                print("[RefAidedeRP1] 警告: normalize=True 会分别缩放 Tx/Rx，"
                      "可能破坏 G~|A|^2A 与 A1 的物理尺度。"
                      "物理功率验证建议使用 normalize=False。")
            pTx = float(np.mean(np.abs(sig2dTx) ** 2))
            pRx = float(np.mean(np.abs(sig2dRx) ** 2))
            self._pAvgTx = pTx
            self._pAvgRx = pRx
            self._lastRunPowerTx = pTx
            self._lastRunPowerRx = pRx

            sig2dTx = sig2dTx * np.sqrt(1.0 / pTx)
            sig2dRx = sig2dRx * np.sqrt(1.0 / pRx)

            if verbose:
                print(f"[RefAidedeRP1] 功率归一化: "
                      f"P_Tx={pTx:.4e} → 1.0, P_Rx={pRx:.4e} → 1.0")
        else:
            self._pAvgTx = pAvgTx if pAvgTx is not None else float(np.mean(np.abs(sig2dTx) ** 2))
            self._pAvgRx = None
            self._lastRunPowerTx = 1.0
            self._lastRunPowerRx = 1.0

        # ── CD Reload 输入信号自动对齐 ───────────────────────────
        if self.afterDSP:
            sig2dTx, sig2dRx = self._alignSignals(sig2dTx, sig2dRx,
                                                  phaseAlign=phaseAlign,
                                                  verbose=verbose)
            self._diagnoseAfterDSPReference(sig2dTx, sig2dRx,
                                            referenceMatched=referenceMatched,
                                            strictReference=strictReference,
                                            verbose=verbose)

        self._lastRunSigTx = sig2dTx.copy()
        self._lastRunSigRx = sig2dRx.copy()

        # ── 构建 G 矩阵 & 求解 γ ─────────────────────────────────
        estBytes = (
            sig2dTx.shape[1] * len(self.zTomoBank) * sig2dTx.shape[0]
            * np.dtype(complex).itemsize
        )
        if verbose and estBytes > 8 * 1024 ** 3:
            print(f"[RefAidedeRP1] 警告: 完整 G 预计需要 {estBytes/1024**3:.1f} GB。")

        G = self.generateMatrixG(sig2dTx, verbose=verbose)
        gammaEst = self.solveGamma(G, sig2dTx, sig2dRx,
                                   lambdaReg=lambdaReg, verbose=verbose)

        # ── 绘图 ─────────────────────────────────────────────────
        if plot:
            self.plotResults(sig2dTx, sig2dRx, gammaEst, G,
                             pAvgTx=self._pAvgTx,
                             interval=interval,
                             allModes=allModes)

        return gammaEst, G

    # ----------------------------------------------------------------
    # 画图方法
    # ----------------------------------------------------------------

    def plotResults(self,
                    sigTx:    NDArray,
                    sigRx:    NDArray,
                    gammaEst: NDArray,
                    G:        NDArray,
                    pAvgTx:   float   = None,
                    interval: NDArray = None,
                    allModes: bool    = True,
                    modeIdx:  int     = 0) -> None:
        """
        绘制层析估计结果图。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plotResults 需要 matplotlib")

        sig2dTx, _ = self._to2d(sigTx)
        sig2dRx, _ = self._to2d(sigRx)
        nModes = sig2dTx.shape[1]

        if pAvgTx is None:
            pAvgTx = float(np.mean(np.abs(sig2dTx) ** 2))
        gammaPlot = gammaEst / self.gamma
        gammaLabel = r'$\gamma$(z) tomo'
        if self._normalizedRun:
            gammaPlot = gammaPlot / pAvgTx
            gammaLabel = r'$\gamma$(z) tomo / $P_{tx}$'

        if interval is None:
            n = sig2dTx.shape[0]
            interval = np.arange(min(320, n // 4), min(800, n // 2))

        Ts = 1.0 / self.Fs
        t  = interval * Ts / 1e-9    # [ns]

        # ── A₀、A[L]、A₁ ──
        A0 = self._linearCD(sig2dTx, length=self.Ltotal)

        if self.afterDSP:
            AL = self.cdReload(sig2dRx)
        else:
            AL = sig2dRx

        # ── 理论 γ(z) ──
        gammaTheory = np.array([
            np.exp(-self.alphaTomo * z +
                   int(np.floor(z / self.Lspan)) * self.Lspan * self.alphaTomo)
            for z in self.zTomoBank
        ])

        # ── 绘制 ──
        modes_to_plot = list(range(nModes)) if allModes else [modeIdx]

        for m in modes_to_plot:
            tx_plot  = sig2dTx[:, m]
            A0_plot  = A0[:, m]
            AL_plot  = AL[:, m]
            A1_plot  = AL_plot - A0_plot
            Gm       = G[m]
            solvedWave = np.dot(Gm.T, gammaEst)

            fig, axes = plt.subplots(4, 1, figsize=(9, 11))
            mode_label = 'pol.X' if m == 0 else ('pol.Y' if m == 1 else f'mode {m}')
            reload_tag = ' [CD Reload]' if self.afterDSP else ''
            fig.suptitle(f'RefAidedeRP1 — Tomography Results ({mode_label}){reload_tag}',
                         fontsize=13)

            # ax[0]: γ(z)
            ax = axes[0]
            ax.plot(self.zTomoBank, gammaPlot, label=gammaLabel)
            ax.plot(self.zTomoBank, gammaTheory,
                    label=r'$\gamma$(z) theory', linestyle='--')
            ax.legend(loc='upper right')
            ax.set_xlabel('Distance (km)')
            ax.xaxis.set_label_position('top')
            ax.set_yscale('log')
            ax.set_title(r'Estimated $\gamma$(z) vs Theory')

            # ax[1]: Tx 时域
            ax = axes[1]
            ax.plot(t, tx_plot[interval].real, 'r-',  label='Tx.re')
            ax.plot(t, tx_plot[interval].imag, 'r--', label='Tx.im')
            ax.legend(loc='upper right')
            ax.set_title(f'Tx Signal (time domain, {mode_label})')
            ax.set_ylabel('Amplitude')

            # ax[2]: A₀ vs A[L]
            ax = axes[2]
            rx_label = 'A[L] (CD Reload)' if self.afterDSP else 'A[L] (fiber output)'
            ax.plot(t, A0_plot[interval].real, 'b-',  label='A₀.re (linear ref)')
            ax.plot(t, A0_plot[interval].imag, 'b--', label='A₀.im')
            ax.plot(t, AL_plot[interval].real, 'r-',  label=f'{rx_label}.re')
            ax.plot(t, AL_plot[interval].imag, 'r--', label=f'{rx_label}.im')
            ax.legend(loc='upper right')
            ax.set_title(f'A₀ vs A[L] ({mode_label})')
            ax.set_ylabel('Amplitude')

            # ax[3]: A₁ vs 层析重建
            ax = axes[3]
            ax.plot(t, A1_plot[interval].real,    'r-',  label='A₁.re (residual)')
            ax.plot(t, A1_plot[interval].imag,    'r--', label='A₁.im (residual)')
            ax.plot(t, solvedWave[interval].real, 'b-',  label='Tomo solve.re')
            ax.plot(t, solvedWave[interval].imag, 'b--', label='Tomo solve.im')
            ax.legend(loc='upper right')
            ax.set_title(f'Residual A₁ vs Tomographic reconstruction ({mode_label})')
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Amplitude')

            plt.tight_layout()

        # ── G 矩阵行可视化 ──
        fig_g, axes_g = plt.subplots(1, nModes, figsize=(6 * nModes, 4),
                                     squeeze=False)
        plotLinePeriod = max(1, len(self.zTomoBank) // 20)

        for m in range(nModes):
            Gm = G[m]
            mode_label = 'pol.X' if m == 0 else ('pol.Y' if m == 1 else f'mode {m}')
            for lineIdx, gRow in enumerate(Gm):
                if lineIdx % plotLinePeriod == 0:
                    axes_g[0, m].plot(t, gRow[interval].real, alpha=0.5)
            axes_g[0, m].set_title(f'G matrix rows (real part, {mode_label})')
            axes_g[0, m].set_xlabel('Time (ns)')
            axes_g[0, m].set_ylabel('Amplitude')

        fig_g.tight_layout()
        plt.show()


class PilotAidedeRP1(RefAidedeRP1):
    """
    Experimental tomography front-end using known high-order Tx data.

    Unlike the conventional pilotWDMRx path, the equalizer training sequence is
    the known full Tx symbol stream, so the 16QAM data can participate in the
    tomography front-end DSP.  Carrier phase is then estimated from phase
    pilots via pilotCPR.  The final tomography waveform can be either compact
    data-only samples (pilot/phase-pilot removed) or one continuous data run.
    """

    def __init__(self, param=None):
        super().__init__(param=param)
        self._lastTomoInfo = None

    def _continuousDataSegment(self, dataMask, nSamples: int,
                               samplesPerSymbol: int, startSymbol: int,
                               trainSymbols, requireFull: bool):
        mask = np.asarray(dataMask, dtype=bool).reshape(-1)
        nSymbolsAvailable = min(len(mask), nSamples // samplesPerSymbol)
        mask = mask[:nSymbolsAvailable]

        targetSymbols = nSymbolsAvailable if trainSymbols is None else int(trainSymbols)
        startSearch = max(0, min(int(startSymbol), nSymbolsAvailable - 1))

        bestStart = None
        bestLen = 0
        runStart = None
        runLen = 0
        for idxSym in range(startSearch, nSymbolsAvailable):
            if mask[idxSym]:
                if runStart is None:
                    runStart = idxSym
                    runLen = 0
                runLen += 1
                if runLen > bestLen:
                    bestStart = runStart
                    bestLen = runLen
                if runLen >= targetSymbols:
                    bestStart = runStart
                    bestLen = runLen
                    break
            else:
                runStart = None
                runLen = 0

        if bestStart is None:
            raise ValueError("[PilotAidedeRP1] dataSymbolMask 中找不到连续 16QAM 数据段。")
        if bestLen < targetSymbols and requireFull:
            raise ValueError("[PilotAidedeRP1] 连续 16QAM 数据段不足："
                             f"需要 {targetSymbols} symbols，找到 {bestLen} symbols。")

        nUseSymbols = min(targetSymbols, bestLen)
        startSample = bestStart * samplesPerSymbol
        stopSample = min(startSample + nUseSymbols * samplesPerSymbol, nSamples)
        return startSample, stopSample, bestStart, nUseSymbols

    def _dataCompactSegment(self, dataMask, nSamples: int,
                            samplesPerSymbol: int, trainSymbols):
        mask = np.asarray(dataMask, dtype=bool).reshape(-1)
        nSymbolsAvailable = min(len(mask), nSamples // samplesPerSymbol)
        mask = mask[:nSymbolsAvailable]
        dataIdx = np.where(mask)[0]
        if trainSymbols is not None:
            dataIdx = dataIdx[:int(trainSymbols)]
        if len(dataIdx) == 0:
            raise ValueError("[PilotAidedeRP1] dataSymbolMask 中没有可用 16QAM 数据符号。")

        sampleIdx = (
            dataIdx[:, np.newaxis] * samplesPerSymbol
            + np.arange(samplesPerSymbol)[np.newaxis, :]
        ).reshape(-1)
        sampleIdx = sampleIdx[sampleIdx < nSamples]
        return sampleIdx, int(dataIdx[0]), int(len(dataIdx))

    def _knownDataEqualizeAndCpr(self, dspResult: dict, param=None,
                                 verbose: bool = True):
        if "rxSyncFO" not in dspResult:
            raise ValueError("[PilotAidedeRP1] dspResult 必须包含 rxSyncFO。")
        if "symbTx" not in dspResult:
            raise ValueError("[PilotAidedeRP1] dspResult 必须包含 symbTx。")
        if "pilotSymbols" not in dspResult:
            raise ValueError("[PilotAidedeRP1] dspResult 必须包含 pilotSymbols。")

        spS = int(getattr(param, 'tomoSpS',
                          max(1, int(round(self.Fs / self.Rs)))))
        rxNorm = pnorm(np.asarray(dspResult["rxSyncFO"]))
        txSymbols = pnorm(np.asarray(dspResult["symbTx"]))

        paramEq = parameters()
        paramEq.SpS = spS
        paramEq.SpSout = int(getattr(param, 'tomoEqSpSout', spS))
        paramEq.nTaps = int(getattr(param, 'tomoEqNTaps', 61))
        paramEq.mu = float(getattr(param, 'tomoEqMu', 1e-3))
        paramEq.nTrain = int(getattr(param, 'tomoEqTrainSymbols',
                                     txSymbols.shape[0]))
        paramEq.dd = bool(getattr(param, 'tomoEqDD', False))
        paramEq.muDD = float(getattr(param, 'tomoEqMuDD', 1e-4))
        paramEq.nTrainDD = int(getattr(param, 'tomoEqTrainDD', 0))
        paramEq.M = int(getattr(param, 'tomoEqM', 16))
        paramEq.verbose = bool(getattr(param, 'tomoEqVerbose', verbose))
        paramEq.pilotMode = 'symbol'

        yEq, W, err = mimoLMSFSE(rxNorm, txSymbols, param=paramEq)

        paramCPR = parameters()
        paramCPR.SpS = paramEq.SpSout
        paramCPR.NSpF = int(getattr(param, 'tomoNSpF',
                                    dspResult.get('NSpF', len(txSymbols))))
        paramCPR.pilotSeq = int(getattr(param, 'tomoPilotSeq',
                                        getattr(param, 'pilotSeq', 2048)))
        paramCPR.phasePilot = getattr(param, 'tomoPhasePilot',
                                      getattr(param, 'phasePilot', 0))
        paramCPR.useInitialPilot = bool(getattr(
            param, 'tomoCprUseInitialPilot',
            paramCPR.phasePilot in (0, None)
        ))
        paramCPR.avgPols = bool(getattr(param, 'tomoCprAvgPols', False))

        if paramCPR.phasePilot in (0, None) and not paramCPR.useInitialPilot:
            raise ValueError("[PilotAidedeRP1] phasePilot=0 时无法只用 phase pilot 做 CPR。")

        yCpr, phiInterp, pilotIdx = pilotCPR(
            yEq, np.asarray(dspResult["pilotSymbols"]), param=paramCPR
        )

        if verbose:
            cpr_src = "phase pilots"
            if paramCPR.useInitialPilot:
                cpr_src = "initial pilotSeq + phase pilots"
            print("[PilotAidedeRP1] known-data pre-DSP:")
            print(f"  EQ training symbols = {paramEq.nTrain} from symbTx")
            print(f"  CPR source = {cpr_src}")

        dspResult['tomo_y_EQ'] = yEq
        dspResult['tomo_y_CPR'] = yCpr
        dspResult['tomo_W'] = W
        dspResult['tomo_err'] = err
        dspResult['tomo_phiInterp'] = phiInterp
        dspResult['tomo_pilotIdx'] = pilotIdx
        return yCpr, W, err, phiInterp, pilotIdx

    def prepareFromDSP(self, dspResult: dict, param=None, verbose: bool = True):
        """
        Build tomography Tx/Rx waveforms from rxSyncFO using known 16QAM data
        and phase-pilot CPR.
        """
        if param is None:
            param = parameters()

        samplesPerSymbol = int(getattr(param, 'tomoSpS',
                                       max(1, int(round(self.Fs / self.Rs)))))
        startSymbol = int(getattr(param, 'tomoStartSymbol', 0))
        trainSymbols = getattr(param, 'tomoTrainSymbols', None)
        requireFull = bool(getattr(param, 'tomoRequireFullDataRun', True))
        doNormalize = bool(getattr(param, 'tomoNormalize', True))
        dataMode = getattr(param, 'tomoDataMode', 'compact')

        if "dataSymbolMask" not in dspResult:
            raise ValueError("[PilotAidedeRP1] dspResult 必须包含 dataSymbolMask。")

        yCpr, _, _, _, _ = self._knownDataEqualizeAndCpr(
            dspResult, param=param, verbose=verbose
        )

        tx = np.asarray(dspResult.get("resampledSigTxMatched",
                                      dspResult["resampledSigTx"]))
        rx = np.asarray(yCpr)
        tx, _ = self._to2d(tx)
        rx, _ = self._to2d(rx)

        nMin = min(tx.shape[0], rx.shape[0])
        tx = tx[:nMin, :]
        rx = rx[:nMin, :]

        if dataMode == 'continuous':
            startSample, stopSample, startSym, nUseSymbols = self._continuousDataSegment(
                dspResult["dataSymbolMask"], nMin, samplesPerSymbol,
                startSymbol, trainSymbols, requireFull
            )
            sigTx = tx[startSample:stopSample, :]
            sigRx = rx[startSample:stopSample, :]
        elif dataMode == 'compact':
            sampleIdx, startSym, nUseSymbols = self._dataCompactSegment(
                dspResult["dataSymbolMask"], nMin, samplesPerSymbol, trainSymbols
            )
            sigTx = tx[sampleIdx, :]
            sigRx = rx[sampleIdx, :]
            startSample = int(sampleIdx[0])
            stopSample = int(sampleIdx[-1] + 1)
        else:
            raise ValueError("[PilotAidedeRP1] tomoDataMode 必须为 'compact' 或 'continuous'。")

        if doNormalize:
            sigTx = pnorm(sigTx)
            sigRx = pnorm(sigRx)

        info = {
            'tomoStartSample': startSample,
            'tomoStopSample': stopSample,
            'tomoStartSymbol': startSym,
            'tomoTrainSymbols': nUseSymbols,
            'tomoSamplesPerSymbol': samplesPerSymbol,
            'tomoDataMode': dataMode,
            'tomoSource': 'known-data FSE + phase-pilot CPR',
        }
        self._lastTomoInfo = info

        if verbose:
            print("[PilotAidedeRP1] tomography data selection:")
            print(f"  mode = {dataMode}, samples = {sigTx.shape[0]} "
                  f"({nUseSymbols} symbols @ {samplesPerSymbol} sps)")
            print(f"  original sample span = [{startSample}:{stopSample}]")
            if dataMode == 'compact':
                print("  note: pilot/phase-pilot samples are removed before G construction")

        return sigTx, sigRx, info

    def run(self, dspResult: dict, param=None, **kwargs) -> tuple:
        if param is None:
            param = parameters()
        if kwargs:
            for key, value in kwargs.items():
                setattr(param, key, value)

        verbose = getattr(param, 'verbose', True)
        plot = getattr(param, 'plot', False)
        lambdaReg = getattr(param, 'lambdaReg', 0)
        normalize = getattr(param, 'normalize', False)
        pAvgTx = getattr(param, 'pAvgTx', None)
        interval = getattr(param, 'interval', None)
        allModes = getattr(param, 'allModes', True)
        referenceMatched = getattr(param, 'referenceMatched', False)
        strictReference = getattr(param, 'strictReference', False)
        fsTx = getattr(param, 'FsTx', None)
        fsRx = getattr(param, 'FsRx', None)

        sig2dTx, sig2dRx, info = self.prepareFromDSP(
            dspResult, param=param, verbose=verbose
        )
        dspResult.update(info)

        if verbose:
            print("[PilotAidedeRP1] tomography options: "
                  "linearCD=edc, pbarCoeff=1.5, solve=augmented")

        if fsTx is not None or fsRx is not None:
            if fsTx is None or fsRx is None:
                raise ValueError("[PilotAidedeRP1] param.FsTx 和 param.FsRx 必须同时提供。")
            if not np.isclose(float(fsTx), float(fsRx), rtol=1e-12, atol=0.0):
                raise ValueError("[PilotAidedeRP1] sigTx/sigRx 采样率不一致: "
                                 f"FsTx={fsTx:.6e}, FsRx={fsRx:.6e}")
            if not np.isclose(float(fsTx), float(self.Fs), rtol=1e-12, atol=0.0):
                raise ValueError("[PilotAidedeRP1] 输入采样率与 param.Fs 不一致: "
                                 f"FsTx=FsRx={fsTx:.6e}, param.Fs={self.Fs:.6e}")

        self._normalizedRun = bool(normalize)
        if normalize:
            pTx = float(np.mean(np.abs(sig2dTx) ** 2))
            pRx = float(np.mean(np.abs(sig2dRx) ** 2))
            self._pAvgTx = pTx
            self._pAvgRx = pRx
            self._lastRunPowerTx = pTx
            self._lastRunPowerRx = pRx
            sig2dTx = sig2dTx * np.sqrt(1.0 / pTx)
            sig2dRx = sig2dRx * np.sqrt(1.0 / pRx)
            if verbose:
                print(f"[PilotAidedeRP1] 功率归一化: "
                      f"P_Tx={pTx:.4e} -> 1.0, P_Rx={pRx:.4e} -> 1.0")
        else:
            self._pAvgTx = pAvgTx if pAvgTx is not None else float(np.mean(np.abs(sig2dTx) ** 2))
            self._pAvgRx = None
            self._lastRunPowerTx = 1.0
            self._lastRunPowerRx = 1.0

        if self.afterDSP and bool(getattr(param, 'diagnoseReference', True)):
            self._diagnoseAfterDSPReference(
                sig2dTx, sig2dRx,
                referenceMatched=referenceMatched,
                strictReference=strictReference,
                verbose=verbose
            )

        self._lastRunSigTx = sig2dTx.copy()
        self._lastRunSigRx = sig2dRx.copy()

        G = self.generateMatrixG(sig2dTx, verbose=verbose)
        gammaEst = self.solveGamma(G, sig2dTx, sig2dRx,
                                   lambdaReg=lambdaReg, verbose=verbose)

        if plot:
            self.plotResults(sig2dTx, sig2dRx, gammaEst, G,
                             pAvgTx=self._pAvgTx,
                             interval=interval,
                             allModes=allModes)

        return gammaEst, G

"""
pilot_wdm_receiver.py
=====================
Pilot-aided WDM coherent receiver with full DSP pipeline.

支持对 WDM 信号中多个波长通道进行轮询解调。

Pipeline（每个通道）
-------------------
    LO → Coherent Rx → MF → Decimation → EDC → Sync → EQ → CPR → Eval

调用方式
--------
::

    from optic.utils import parameters

    paramRx = parameters()
    paramRx.paramCh     = paramCh          # 必须：信道参数（Ltotal, D, Fc）
    paramRx.FO          = -128e6           # 激光器频偏 [Hz]
    paramRx.polRotation = np.pi / 3        # 偏振旋转角 [rad]
    paramRx.channels    = 'all'            # 'all' / 'center' / [0,1,2]

    # EQ
    paramRx.nTaps   = 61
    paramRx.mu      = 1e-3
    paramRx.dd      = False
    paramRx.eqSpSout = 1                   # EQ 输出采样率（1=符号率）

    # CPR
    paramRx.cprN    = 35
    paramRx.cprB    = 64

    receiver = PilotWDMReceiver(transmitter, sigWDM, paramRx)
    results  = receiver.run(plot=False, verbose=True)

    # 查看某通道结果
    r = results[1]              # 通道 #1
    y_CPR = r['y_CPR']          # CPR 输出
    BER   = r['BER']            # 误码率
    NGMI  = r['NGMI']           # 归一化 GMI

    # 汇总表
    receiver.printSummary()

WDM 通道轮询
-------------
``channels`` 参数控制处理哪些通道：

- ``'center'``（默认）：只处理中心通道 ``nChannels // 2``
- ``'all'``：轮询所有 ``nChannels`` 个通道
- ``[0, 2]``：只处理指定编号的通道

各通道共享 LO / 前端 / EQ / CPR 参数，但独立执行完整 DSP。
通道之间结果互不影响。

与层析管线联用（方案 B）
------------------------
::

    paramRx.eqSpSout = 2      # EQ 保持 2 采样/符号
    paramRx.cprSpS   = 2      # CPR 处理 2 采样/符号
    receiver.run()

    # 取出 CPR 输出 → CD Reload → 层析
    r = receiver.results[chIndex]
    y_CPR2 = r['y_CPR']                   # shape (N, 2), SpS=2
    resampledSigTx = r['resampledSigTx']  # shape (N, 2), SpS=2

    paramTomo.afterDSP = True
    ppe = RefAidedeRP1(paramTomo)
    ppe.run(resampledSigTx, y_CPR2, ...)
"""


class pilotWDMRx:
    """
    Pilot-aided WDM coherent receiver with full DSP pipeline.

    Parameters
    ----------
    transmitter : pilotWDMTx
        Transmitter object (from ``pilotWDMTx(paramTx)``).
    sigWDM : ndarray, shape (nSamples, nModes)
        Received WDM optical field after fiber propagation.
    param : parameters
        Receiver configuration. See below for all fields.

    Attributes
    ----------
    results : dict
        ``{chIndex: dict}`` with per-channel DSP results.
        Each channel dict contains:

        =========== ====================================================
        Key         Description
        =========== ====================================================
        y_EQ        Equalizer output, shape ``(nSymb, nModes)``
        y_CPR       CPR output, shape ``(nSymb, nModes)``
        W           Converged MIMO filter weights
        err         EQ error vector
        BER         Bit error rate per mode
        SER         Symbol error rate per mode
        SNR         SNR per mode [dB]
        GMI         Generalized mutual information per mode [bits]
        NGMI        Normalized GMI per mode
        delay       Synchronization delay [samples]
        fo_est      Estimated frequency offset [Hz]
        pilotSymbols   Normalized pilot symbols used
        resampledSigTx Decimated Tx waveform (for tomography)
        resampledPilot Decimated pilot waveform
        =========== ====================================================

    param fields
    ------------
    Required
    ^^^^^^^^
    paramCh : parameters
        Channel parameters object containing at least:
        ``Ltotal`` (km), ``D`` (ps/nm/km), ``Fc`` (Hz).

    Optional (with defaults)
    ^^^^^^^^^^^^^^^^^^^^^^^^

    *Channel selection:*

    channels : list of int or str, default ``'center'``
        ``'center'``: center channel only.
        ``'all'``: all channels.
        ``[0, 1, 2]``: specific channel indices.

    *Frequency offset:*

    FO : float, default 0
        Laser frequency offset [Hz].

    *LO (Local Oscillator):*

    loP : float, default 10
        LO power [dBm].
    loLw : float, default 100e3
        LO linewidth [Hz].
    loRINvar : float, default 0
        LO RIN variance.
    loSeed : int, default 789
        LO random seed (offset by +chIndex per channel).

    *Frontend impairments:*

    polRotation : float, default 0
        Input polarization rotation [rad].
    pdl : float, default 0
        Polarization dependent loss [dB].
    polDelay : float, default 0
        Polarization delay [s].
    phaseImbX, phaseImbY : float, default 0
        IQ phase imbalance X/Y [rad].
    ampImbX, ampImbY : float, default 0
        IQ amplitude imbalance X/Y [dB].

    *Photodiode:*

    pdIdeal : bool, default True
        Use ideal photodiode.
    pdSeed : int, default 1011
        PD random seed.

    *Decimation:*

    SpSout : int, default 2
        Output samples per symbol after decimation.

    *Equalization:*

    nTaps : int, default 61
        MIMO FIR filter length.
    mu : float, default 1e-3
        LMS step size (pilot phase).
    nTrain : int, default ``pilotSeq``
        Number of pilot symbols for supervised training.
    dd : bool, default False
        Enable decision-directed (DD) refinement.
    muDD : float, default 1e-4
        DD step size.
    nTrainDD : int, default 20000
        DD training symbols.
    eqSpSout : int, default 1
        EQ output samples per symbol (1=symbol, 2=sample domain).
    pilotMode : str, default ``'symbol'``
        ``'symbol'``: pilot is compact symbol array.
        ``'sample'``: pilot is sample-rate waveform.

    *Carrier phase recovery:*

    cprN : int, default 35
        BPS test angles.
    cprB : int, default 64
        BPS block length.
    cprSpS : int, default 1
        CPR input SpS (1=symbol, 2=sample domain).

    *Synchronization:*

    syncSearchMode : str, default ``'best'``
        ``'best'``: joint search across modes.
        ``'perMode'``: independent per-mode search.

    *Evaluation:*

    evalDiscard : int, default 500
        Discard edge symbols before evaluation.
    """

    def __init__(self, transmitter, sigWDM, param=None):
        if param is None:
            param = parameters()

        self.transmitter = transmitter
        self.sigWDM = sigWDM

        # ── 从 transmitter 自动推导 ──────────────────────────────────
        txp = transmitter.param
        self.nChannels   = txp.nChannels
        self.nModes      = txp.nPolModes
        self.SpS         = txp.SpS
        self.Rs          = txp.Rs
        self.Fs          = self.Rs * self.SpS
        self.M           = txp.M
        self.Fc          = txp.Fc
        self.freqGrid    = transmitter.freqGrid
        self.pilotSeq    = txp.pilotSeq
        self.phasePilot  = getattr(txp, 'phasePilot', 0)
        self.pilotM      = getattr(txp, 'pilotM', 4)
        self.NSpF        = transmitter.NSpF

        # ── 必填参数 ────────────────────────────────────────────────
        if not hasattr(param, 'paramCh'):
            raise ValueError(
                "[PilotWDMReceiver] param.paramCh 为必填参数（需包含 Ltotal, D, Fc）"
            )
        self.paramCh = param.paramCh

        # ── 通道选择 ────────────────────────────────────────────────
        channels = getattr(param, 'channels', 'center')
        if channels == 'center' or channels is None:
            self.channels = [int(np.floor(self.nChannels / 2))]
        elif channels == 'all':
            self.channels = list(range(self.nChannels))
        else:
            self.channels = list(channels)

        # ── 频偏 ────────────────────────────────────────────────────
        self.FO = getattr(param, 'FO', 0)

        # ── LO ──────────────────────────────────────────────────────
        self.loP      = getattr(param, 'loP', 10)
        self.loLw     = getattr(param, 'loLw', 100e3)
        self.loRINvar = getattr(param, 'loRINvar', 0)
        self.loSeed   = getattr(param, 'loSeed', 789)

        # ── 前端 ────────────────────────────────────────────────────
        self.polRotation = getattr(param, 'polRotation', 0)
        self.pdl         = getattr(param, 'pdl', 0)
        self.polDelay    = getattr(param, 'polDelay', 0)
        self.phaseImbX   = getattr(param, 'phaseImbX', 0)
        self.phaseImbY   = getattr(param, 'phaseImbY', 0)
        self.ampImbX     = getattr(param, 'ampImbX', 0)
        self.ampImbY     = getattr(param, 'ampImbY', 0)

        # ── PD ──────────────────────────────────────────────────────
        self.pdIdeal = getattr(param, 'pdIdeal', True)
        self.pdSeed  = getattr(param, 'pdSeed', 1011)

        # ── 下采样 ──────────────────────────────────────────────────
        self.SpSout = getattr(param, 'SpSout', 2)

        # ── EQ ──────────────────────────────────────────────────────
        self.nTaps     = getattr(param, 'nTaps', 61)
        self.mu        = getattr(param, 'mu', 1e-3)
        self.nTrain    = getattr(param, 'nTrain', self.pilotSeq)
        self.dd        = getattr(param, 'dd', False)
        self.muDD      = getattr(param, 'muDD', 1e-4)
        self.nTrainDD  = getattr(param, 'nTrainDD', 20000)
        self.eqM       = getattr(param, 'eqM', self.M)
        self.eqVerbose = getattr(param, 'eqVerbose', True)
        self.eqSpSout  = getattr(param, 'eqSpSout', 1)
        self.pilotMode = getattr(param, 'pilotMode', 'symbol')

        # ── CPR ─────────────────────────────────────────────────────
        self.cprN   = getattr(param, 'cprN', 35)
        self.cprB   = getattr(param, 'cprB', 64)
        self.cprSpS = getattr(param, 'cprSpS', 1)

        # ── Sync ────────────────────────────────────────────────────
        self.syncSearchMode = getattr(param, 'syncSearchMode', 'best')

        # ── Eval ────────────────────────────────────────────────────
        self.evalDiscard = getattr(param, 'evalDiscard', 500)

        # ── 结果 ────────────────────────────────────────────────────
        self.results = {}

    # ═══════════════════════════════════════════════════════════════
    #  公共方法
    # ═══════════════════════════════════════════════════════════════

    def run(self, channels=None, plot=False, verbose=True):
        """
        Run the full DSP pipeline for selected WDM channels.

        Parameters
        ----------
        channels : list of int, int, str, or None
            Channels to process. ``None`` uses ``self.channels`` (from param).
            Can pass a single int, ``'all'``, or a list of indices.
        plot : bool, default False
            If True, plot constellation after CPR for each channel.
        verbose : bool, default True
            Print per-step timing and metrics.

        Returns
        -------
        results : dict
            ``{chIndex: dict}`` with per-channel results.
        """
        if channels is None:
            channels = self.channels
        elif isinstance(channels, int):
            channels = [channels]
        elif channels == 'all':
            channels = list(range(self.nChannels))

        for ch in channels:
            if ch < 0 or ch >= self.nChannels:
                raise ValueError(
                    f"[PilotWDMReceiver] 通道 #{ch} 超出范围 [0, {self.nChannels - 1}]"
                )
            self._processChannel(ch, plot=plot, verbose=verbose)

        if verbose and len(channels) > 1:
            self.printSummary()

        return self.results

    def printSummary(self):
        """
        Print a summary table of BER / SNR / NGMI for all processed channels.
        """
        if not self.results:
            print("[PilotWDMReceiver] 尚未处理任何通道。")
            return

        nModes = self.nModes
        polLabels = ['X', 'Y'] if nModes == 2 else [str(i) for i in range(nModes)]

        # Header
        header = f"{'Ch':>4s}"
        for p in polLabels:
            header += f"  {'BER_'+p:>10s}  {'SNR_'+p:>8s}  {'NGMI_'+p:>7s}"
        print('\n' + '=' * len(header))
        print('  PilotWDMReceiver — Summary')
        print('=' * len(header))
        print(header)
        print('-' * len(header))

        for ch in sorted(self.results.keys()):
            r = self.results[ch]
            row = f"  {ch:>2d}"
            for m in range(nModes):
                row += f"  {r['BER'][m]:>10.2e}  {r['SNR'][m]:>7.2f} dB  {r['NGMI'][m]:>6.2f}"
            print(row)

        print('=' * len(header))

    def plotConstellation(self, chIndex, stage='cpr', R=1.5):
        """
        Plot constellation diagram for a processed channel.

        Parameters
        ----------
        chIndex : int
            Channel index.
        stage : str
            ``'eq'`` or ``'cpr'`` (default).
        R : float
            Plot radius.
        """
        if chIndex not in self.results:
            raise ValueError(f"通道 #{chIndex} 尚未处理。")

        r = self.results[chIndex]
        d = self.evalDiscard

        if stage == 'eq':
            sig = r['y_EQ']
            title = f'Ch #{chIndex} — After EQ'
        elif stage == 'cpr':
            sig = r['y_CPR']
            title = f'Ch #{chIndex} — After CPR'
        else:
            raise ValueError(f"stage 必须为 'eq' 或 'cpr'，收到 '{stage}'")

        pconst(sig[d:-d, :] if d > 0 else sig, R=R)

    # ═══════════════════════════════════════════════════════════════
    #  DSP 子步骤（可被子类 override）
    # ═══════════════════════════════════════════════════════════════

    def _generateLO(self, chIndex):
        """Generate local oscillator field for a given channel."""
        paramLO = parameters()
        paramLO.P         = self.loP
        paramLO.lw        = self.loLw
        paramLO.RIN_var   = self.loRINvar
        paramLO.Ns        = len(self.sigWDM)
        paramLO.Fs        = self.Fs
        paramLO.seed      = self.loSeed + chIndex
        paramLO.freqShift = self.freqGrid[chIndex] + self.FO
        return basicLaserModel(paramLO)

    def _coherentDetection(self, sigLO):
        """Perform polarization-multiplexed coherent detection."""
        paramFE = parameters()
        paramFE.Fs          = self.Fs
        paramFE.polRotation = self.polRotation
        paramFE.pdl         = self.pdl
        paramFE.polDelay    = self.polDelay
        paramFE.phaseImbX   = self.phaseImbX
        paramFE.phaseImbY   = self.phaseImbY
        paramFE.ampImbX     = self.ampImbX
        paramFE.ampImbY     = self.ampImbY

        paramPD = parameters()
        paramPD.B     = self.Rs
        paramPD.Fs    = self.Fs
        paramPD.ideal = self.pdIdeal
        paramPD.seed  = self.pdSeed

        return pdmCoherentReceiver(self.sigWDM, sigLO, paramFE, paramPD)

    def _matchedFilter(self, sigRx):
        """Apply matched filter (pulse shaping convolution)."""
        txp = self.transmitter.param
        paramPS = parameters()
        paramPS.SpS         = self.SpS
        paramPS.nFilterTaps = txp.nFilterTaps
        paramPS.rollOff     = txp.pulseRollOff
        paramPS.pulseType   = txp.pulseType
        pulse = pulseShape(paramPS)
        return firFilter(pulse, sigRx)

    def _decimate(self, sigRx, chIndex):
        """
        Downsample Rx and Tx signals from SpS to SpSout.

        Returns
        -------
        sigRx : ndarray
            Decimated Rx signal.
        pilotTemplate : ndarray
            Decimated pilot preamble (for sync).
        resampledPilot : ndarray
            Decimated full pilot waveform.
        resampledSigTx : ndarray
            Decimated Tx waveform (for tomography).
        """
        paramDec = parameters()
        paramDec.SpSin  = self.SpS
        paramDec.SpSout = self.SpSout

        sigRx = decimate(sigRx, paramDec)

        pulsePilot    = self.transmitter.pulseSampledPilot[:, :, chIndex]
        pulseTraining = pulsePilot[:self.pilotSeq * self.SpS, :]
        pilotTemplate = decimate(pulseTraining, paramDec)

        resampledPilot = decimate(pulsePilot, paramDec)
        txPulse = self.transmitter.pulseTxWDM[:, :, chIndex]
        resampledSigTx = decimate(txPulse, paramDec)
        resampledSigTxMatched = decimate(self._matchedFilter(txPulse), paramDec)
        idxFrame, idxDataFrame, idxPilotFrame = self.transmitter._cal_pilot_idx(
            self.transmitter.param
        )
        dataSymbolMask = np.tile(idxDataFrame, self.transmitter.param.nFrames)
        pilotSymbolMask = np.tile(idxPilotFrame, self.transmitter.param.nFrames)

        return (
            sigRx, pilotTemplate, resampledPilot, resampledSigTx,
            resampledSigTxMatched, dataSymbolMask, pilotSymbolMask
        )

    def _cdCompensation(self, sigRx):
        """Electronic dispersion compensation (EDC)."""
        paramEDC = parameters()
        paramEDC.L  = self.paramCh.Ltotal
        paramEDC.D  = self.paramCh.D
        paramEDC.Fc = self.paramCh.Fc
        paramEDC.Rs = self.Rs
        paramEDC.Fs = self.SpSout * self.Rs
        return edc(sigRx, paramEDC)

    def _synchronize(self, sigRx, pilotTemplate):
        """Pilot-based symbol synchronization + frequency offset estimation."""
        paramSync = parameters()
        paramSync.SpS        = self.SpSout
        paramSync.Fs         = self.Rs * self.SpSout
        paramSync.pilotM     = self.pilotM
        paramSync.searchMode = self.syncSearchMode
        paramSync.NSpF       = self.NSpF
        return pilotBasedSync(sigRx, pilotTemplate, param=paramSync)

    def _equalize(self, rxNorm, pilot):
        """MIMO LMS FSE equalization (pilot + optional DD)."""
        paramEq = parameters()
        paramEq.SpS       = self.SpSout
        paramEq.SpSout    = self.eqSpSout
        paramEq.nTaps     = self.nTaps
        paramEq.mu        = self.mu
        paramEq.nTrain    = self.nTrain
        paramEq.dd        = self.dd
        paramEq.muDD      = self.muDD
        paramEq.nTrainDD  = self.nTrainDD
        paramEq.M         = self.eqM
        paramEq.verbose   = self.eqVerbose
        paramEq.pilotMode = self.pilotMode
        return mimoLMSFSE(rxNorm, pilot, param=paramEq)

    def _cpr(self, y_EQ, pilotSymbols):
        """Pilot-aided carrier phase recovery."""
        paramCPR = parameters()
        paramCPR.alg          = 'bps'
        paramCPR.M            = self.M
        paramCPR.N            = self.cprN
        paramCPR.B            = self.cprB
        paramCPR.returnPhases = False
        paramCPR.Ts           = 1 / self.Rs
        paramCPR.NSpF         = self.NSpF
        paramCPR.pilotSeq     = self.pilotSeq
        paramCPR.phasePilot   = self.phasePilot
        paramCPR.SpS          = self.cprSpS
        return pilotCPR(y_EQ, pilotSymbols, param=paramCPR)

    def _evaluate(self, y_CPR, symbTx):
        """Compute BER, SER, SNR, GMI, NGMI."""
        d = self.evalDiscard
        if self.cprSpS > 1:
            y_Metrics = y_CPR[::self.cprSpS, :]
        else:
            y_Metrics = y_CPR

        nAligned = min(y_Metrics.shape[0], symbTx.shape[0])
        symbTxAligned = symbTx[:nAligned, :]
        y_Eval        = y_Metrics[:nAligned, :]

        # 可选丢弃边缘符号（用于 eval 计算）
        if d > 0 and nAligned > 2 * d:
            yE = y_Eval[d:-d, :].copy()
            sE = symbTxAligned[d:-d, :].copy()
        else:
            yE = y_Eval.copy()
            sE = symbTxAligned.copy()

        BER, SER, SNR = fastBERcalc(yE, sE, self.M, 'qam')
        GMI, NGMI     = monteCarloGMI(yE, sE, self.M, 'qam')

        return {
            'BER': BER, 'SER': SER, 'SNR': SNR,
            'GMI': GMI, 'NGMI': NGMI,
            'y_Eval': y_Eval, 'symbTxAligned': symbTxAligned,
        }

    # ═══════════════════════════════════════════════════════════════
    #  通道处理主循环
    # ═══════════════════════════════════════════════════════════════

    def _processChannel(self, chIndex, plot=False, verbose=True):
        """
        Run the complete DSP pipeline for a single WDM channel.

        Parameters
        ----------
        chIndex : int
            WDM channel index (0-based).
        plot : bool
            Plot constellation after CPR.
        verbose : bool
            Print per-step timing and metrics.

        Returns
        -------
        result : dict
            Channel result dictionary, also stored in ``self.results[chIndex]``.
        """
        t0 = time.time()
        fc_hz = self.Fc + self.freqGrid[chIndex]

        if verbose:
            print(f'\n{"=" * 65}')
            print(f'[PilotWDMReceiver] Channel #{chIndex}/{self.nChannels - 1}  '
                  f'fc={fc_hz / 1e12:.4f} THz  '
                  f'λ={const.c / fc_hz / 1e-9:.4f} nm')
            print(f'{"=" * 65}')

        # ── 1. LO ───────────────────────────────────────────────────
        sigLO = self._generateLO(chIndex)
        if verbose:
            print(f'  [1/7] LO: P={self.loP} dBm, lw={self.loLw / 1e3:.1f} kHz, '
                  f'FO={self.FO / 1e6:.1f} MHz')

        # ── 2. Coherent detection ───────────────────────────────────
        sigRx = self._coherentDetection(sigLO)
        if verbose:
            print(f'  [2/7] Coherent Rx: shape {sigRx.shape}')

        # ── 3. Matched filter ───────────────────────────────────────
        t1 = time.time()
        sigRx = self._matchedFilter(sigRx)
        if verbose:
            print(f'  [3/7] Matched filter: {time.time() - t1:.2f}s')

        # ── 4. Decimation ───────────────────────────────────────────
        t1 = time.time()
        (sigRx, pilotTemplate, resampledPilot, resampledSigTx,
         resampledSigTxMatched, dataSymbolMask, pilotSymbolMask) = \
            self._decimate(sigRx, chIndex)
        if verbose:
            print(f'  [4/7] Decimation ({self.SpS}→{self.SpSout}): '
                  f'{time.time() - t1:.2f}s, shape {sigRx.shape}')

        # ── 5. CD compensation ──────────────────────────────────────
        t1 = time.time()
        sigRx = self._cdCompensation(sigRx)
        if verbose:
            print(f'  [5/7] EDC ({self.paramCh.Ltotal} km): {time.time() - t1:.2f}s')

        # ── 6. Synchronization ──────────────────────────────────────
        t1 = time.time()
        rxSyncFO, delay, fo_est = self._synchronize(sigRx, pilotTemplate)
        if verbose:
            print(f'  [6/7] Sync: delay={delay}, FO_est={fo_est / 1e6:.2f} MHz, '
                  f'{time.time() - t1:.2f}s')

        # ── 7. Normalization + EQ + CPR ─────────────────────────────
        rxNorm       = pnorm(rxSyncFO)
        pilotSymbols = pnorm(self.transmitter.pilot[:, :, chIndex])

        # 7a. Equalization
        y_EQ, W, err = self._equalize(rxNorm, pilotSymbols)

        # 7b. CPR
        t1 = time.time()
        y_CPR, phiInterp, pilotIdx = self._cpr(y_EQ, pilotSymbols)
        if verbose:
            print(f'  [7/7] CPR: {time.time() - t1:.2f}s')

        # ── 8. Evaluation ───────────────────────────────────────────
        symbTx  = self.transmitter.symbTxWDM[:, :, chIndex]
        metrics = self._evaluate(y_CPR, symbTx)

        totalTime = time.time() - t0

        if verbose:
            nM = self.nModes
            polL = ['pol.X', 'pol.Y'] if nM == 2 else [f'mode.{i}' for i in range(nM)]
            header = '      ' + '      '.join(f'{l:>8s}' for l in polL)
            print(header)
            print(' SER:', ',  '.join(f'{metrics["SER"][m]:.2e}' for m in range(nM)))
            print(' BER:', ',  '.join(f'{metrics["BER"][m]:.2e}' for m in range(nM)))
            print(' SNR:', ',  '.join(f'{metrics["SNR"][m]:.2f} dB' for m in range(nM)))
            print(' GMI:', ',  '.join(f'{metrics["GMI"][m]:.2f} bits' for m in range(nM)))
            print('NGMI:', ',  '.join(f'{metrics["NGMI"][m]:.2f}' for m in range(nM)))
            print(f'  ── Total: {totalTime:.2f}s')

        # ── Plot ────────────────────────────────────────────────────
        if plot:
            d = self.evalDiscard
            sig = y_CPR[d:-d, :] if d > 0 and y_CPR.shape[0] > 2 * d else y_CPR
            pconst(sig, R=1.5)

        # ── Store ───────────────────────────────────────────────────
        result = {
            # Intermediate signals
            'rxSyncFO':       rxSyncFO,
            'rxNorm':         rxNorm,
            'y_EQ':           y_EQ,
            'y_CPR':          y_CPR,
            'W':              W,
            'err':            err,
            'phiInterp':      phiInterp,
            'pilotIdx':       pilotIdx,
            # Sync info
            'delay':          delay,
            'fo_est':         fo_est,
            # Tx references (for tomography)
            'pilotTemplate':    pilotTemplate,
            'resampledPilot':   resampledPilot,
            'resampledSigTx':   resampledSigTx,
            'resampledSigTxMatched': resampledSigTxMatched,
            'dataSymbolMask':   dataSymbolMask,
            'pilotSymbolMask':  pilotSymbolMask,
            'pilotSymbols':     pilotSymbols,
            'symbTx':           symbTx,
            # Metrics
            **metrics,
        }

        self.results[chIndex] = result
        return result

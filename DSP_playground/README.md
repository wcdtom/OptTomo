### 动手动脚学DSP
* 从OptiCommPy例子总结各个模块算法
  * physical model
  * optic.dsp.{core, 信道均衡 --> 时钟恢复 --> 载波恢复}
  * 甚至包含数字反传模块
> NOTE: OptiCommPy代码中包含许多希腊字母作为变量名，但是这真的能运行

#### test.WDM_amp_transmission
* 参数设置：Tx, channel, EDFA
* paramTx --> **simpleWDMTx** --> sigWDM_Tx --> **manakovSSF**
--> sigWDM --> **edfaSM** --> sigWDM_Amp --> **pdmCoherentReceiver**(添加固定的polarization rotation angle)
--> sigRx(这里开始不同模分别进行)--> **firFilter** --> **edc**(色散补偿) --> **decimate**(重采样) -->
**symbolSync**(时钟恢复) --> **mimoAdaptEqualizer**(偏振解复用+信道均衡) --> **cpr**(载波恢复:频偏，相偏) --> 
生成统计数据：
  * fastBERcalc: BER(Bit-error-rate), SER(Symbol-error-rate), SNR
  * monteCarloGMI: GMI(Generalized mutual information), NGMI(Normalized mutual information)
  * monteCarloMI: MI(Monte Carlo based mutual information)
  * calcEVM: EVM(error vector magnitude)

#### Tx: a simple WDM transmitter
* 生成多个波长，多偏振模式的信号: 
```python
sigTxWDM = np.zeros((len(t), param.Nmodes), dtype="complex")
# 不同波长的信号是加在一起的
for indCh in tqdm(range(param.Nch)):
    for indMode in range(param.Nmodes):
        sigTxWDM[:, indMode] += sigTxCh * np.exp(1j * 2 * π * (freqGrid[indCh] / Fs) * t)
```
* np.random --> **modulateGray** --> upsample --> firFilter
--> (if indMode==0) phaseNoise --> **iqm** --> sigTxCh
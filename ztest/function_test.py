from optic_plus.dsp_plus.core_plus import pulseShape_plus

filterCoeffs = pulseShape_plus(pulseType='sinc',
                               SpS=16,
                               N=2048,
                               alpha=0,
                               Ts=1e-9)

# from optic_plus.model_plus.tx_plus import pilotWDMTx

# transmitter = pilotWDMTx()
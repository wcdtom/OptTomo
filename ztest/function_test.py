from optic_plus.dsp_plus.core_plus import pulseShape_plus

filterCoeffs = pulseShape_plus(pulseType='sinc',
                               SpS=16,
                               N=2048,
                               alpha=0.6,
                               Ts=1e-9)

# from optic_plus.model_plus.tx_plus import pilotWDMTx

# transmitter = pilotWDMTx()

# if param.pulse == "nrz":
#    pulse = pulseShape("nrz", param.SpS)
# elif param.pulse == "rrc":
#    pulse = pulseShape("rrc", param.SpS, N=param.Ntaps, alpha=param.alphaRRC, Ts=Ts)

#pulse = pulse / np.max(np.abs(pulse))
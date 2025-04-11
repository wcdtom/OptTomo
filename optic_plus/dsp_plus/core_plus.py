import numpy as np
import matplotlib.pyplot as plt


def pulseShape_plus(pulseType, SpS=16, N=2048, alpha=0.6, Ts=1e-9):
    """
    Generate a pulse shaping filter.

    Parameters
    ----------
    pulseType : string ('rect','nrz','rrc')
        type of pulse shaping filter.
    SpS : int, optional
        Number of samples per symbol of input signal. The default is 2.
    N : int, optional
        Number of filter coefficients. The default is 1024.
    alpha : float, optional
        Rolloff of RRC filter. The default is 0.1.
    Ts : float, optional
        Symbol period in seconds. The default is 1.

    Returns
    -------
    filterCoeffs : np.array
        Array of filter coefficients (normalized).

    """
    fa = (1 / Ts) * SpS

    if pulseType == "sinc":
        t = np.linspace(-N // 2, N // 2, N) * (1 / fa)
        filterCoeffs = sincFilterTaps(t, alpha, Ts)
    else:
        filterCoeffs = None

    filterCoeffs = filterCoeffs / np.sqrt(np.sum(filterCoeffs**2))
    # the first value in filterCoeffs is zero
    # for test
    # print(filterCoeffs)
    # fig, bx = plt.subplots()
    # t_ = np.linspace((-N // 2)-1, N // 2, N) * (1 / fa)
    # bx.set(xlim=(-Ts/2, Ts/2), xlabel="t")
    # bx.plot(t_, np.real(filterCoeffs), linewidth=2, label=r"$a(t)$")
    # bx.legend(fontsize=14)
    # plt.show()
    return filterCoeffs


def sincFilterTaps(t, beta, Ts):
    """
    Generate Nyquist filter coefficients.

    Parameters
    ----------
    t : np.array
        Time values.
    beta : float [0, 1]
        Nyquist filter roll-off factor (in frequency domain).
    Ts : float
        Symbol period.

    Returns
    -------
    coeffs : np.array
        Nyquist filter coefficients.

    References
    ----------
    [1] Hui, R. (2020). Introduction to Fiber-Optic Communications (Chapter 11). Springer.
    """
    boundary = ((1-beta)/(2*Ts), (1+beta)/(2*Ts))
    f = np.linspace(-1/Ts, 1/Ts, len(t))
    H_f = np.zeros(len(t), dtype=np.float64)

    for i, f_i in enumerate(f):
        if np.abs(f_i) <= boundary[0]:
            H_f[i] = (np.pi*f_i*Ts)/np.sin(np.pi*f_i*Ts)
        elif np.abs(f_i) > boundary[0] and np.abs(f_i) < boundary[1]:
            H_f[i] = ((np.pi*f_i*Ts)/np.sin(np.pi*f_i*Ts)) * np.cos(((np.pi*Ts)/(2*beta))*(np.abs(f_i)-boundary[0]))
        elif np.abs(f_i) >= boundary[1]:
            H_f[i] = 0
    # for test
    # fig, ax = plt.subplots()
    # _f = f*Ts
    # ax.plot(_f, H_f, linewidth=2, label=r"$H(f)$")
    # ax.set(xlim=(-1, 1), xlabel="f (GHz)")
    # ax.legend(fontsize=14)
    # plt.show()
    coeffs = np.fft.ifftshift(np.fft.ifft(H_f, len(t)))

    return coeffs
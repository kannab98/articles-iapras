import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import pi
from seawave.spectrum import spectrum
from matplotlib.backends.backend_pdf import PdfPages
from seawave import rc

plt.style.use(['science', 'ieee'])


U  = [5, 10, 15]
X  = [3000, 8000, 15000, 20170]





with PdfPages('spectrum.pdf') as pdf:
    fig, ax = plt.subplots()
    for u in U:
        rc.wind.speed = u 
        spectrum.__call__dispatcher__(radar_dispatcher=False)
        k = spectrum.k
        S = spectrum(k, dispatcher=False, radar_dispatcher=False)
        ax.loglog(k, S, label="$U_{10} = %d$ м/с" % u)

    ax.set_xlabel("$\\kappa, \\text{рад} \\cdot \\text{ м}^{-1}$")
    ax.set_ylabel("$S(\\kappa), \\text{м}^3 \\cdot \\text{ рад}^{-1}$")
    ax.legend()
    pdf.savefig()

    fig, ax = plt.subplots()
    rc.wind.speed = 5
    for x in X:
        rc.surface.nonDimWindFetch = x 
        spectrum.__call__dispatcher__(radar_dispatcher=False)
        k = spectrum.k
        S = spectrum(k, dispatcher=False, radar_dispatcher=False)
        ax.loglog(k, S, label="$\\tilde x = %d$" % x)

    ax.set_xlabel("$\\kappa, \\text{рад} \\cdot \\text{ м}^{-1}$")
    ax.set_ylabel("$S(\\kappa), \\text{м}^3 \\cdot \\text{ рад}^{-1}$")
    ax.legend()
    pdf.savefig()
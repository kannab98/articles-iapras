import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from seawave.spectrum import spectrum
from seawave.surface import surface
from seawave import cuda, kernel, rc




plt.style.use(['science', 'ieee'])
x, y = surface.meshgrid
srf = kernel.launch(cuda.default)


from matplotlib.backends.backend_pdf import PdfPages

N = len(rc.antenna.waveLength)
with PdfPages('surfaces_bounds.pdf') as pdf:


    for i in range(3):
        fig, ax = plt.subplots()
        ylabel = ["$\\zeta, \\text{ м}$", "$\\xi_{xx}$", "$\\xi_{yy}$"]

        for n  in range(N, 0, -1):
            arr = kernel.convert_to_band(srf, n)
            ax.plot(x, arr.iloc[:, i], label=rc.antenna.band[n-1])
            ax.set_xlabel("$x, \\text{ м}$")
            ax.set_ylabel(ylabel[i])

        plt.legend()
        pdf.savefig(figure=fig)

    for i in range(3):
        fig, ax = plt.subplots()
        ylabel = ["$\\Delta \\zeta, \\text{ м}$", "$\\Delta \\xi_{xx}$", "$\\Delta \\xi_{yy}$"]

        arr0 = kernel.convert_to_band(srf, 1)
        for n  in range(N, 1, -1):
            arr = kernel.convert_to_band(srf, n)
            ax.plot(x, arr0.iloc[:, i] - arr.iloc[:, i], label='%s - %s' % (rc.antenna.band[0], rc.antenna.band[n-1]))

        ax.set_xlabel("$x, \\text{ м}$")
        ax.set_ylabel(ylabel[i])
        plt.legend()
        pdf.savefig(figure=fig)






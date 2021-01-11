from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np

srf1 = kernel.simple_launch(cuda.cwm)
srf0 = kernel.simple_launch(cuda.default)
x = surface.gridx

figs = []
axs = []
lbly = ["$\\zeta(x)$, м", "$\\sigma_x(x)$, м", "$\\sigma_y(x)$, м",
         "$v_z(x)$, м", "$v_x(x)$, м", "$v_y(x)$, м"]

nms = ["h", "sigmax", "sigmay", "vz", "vx", "vy"]

for i in range(len(nms)):
    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)
    axs[-1].plot(x, srf1[i])
    axs[-1].plot(x, srf0[i])
    axs[-1].set_xlabel("$x$, м")
    axs[-1].set_ylabel(lbly[i])

    # figs[-1].savefig(nms[i]+'.png')


import matplotlib.backends.backend_pdf
with matplotlib.backends.backend_pdf.PdfPages("cwm_demo_1d.pdf") as pdf:
    for fig in figs:
        pdf.savefig( fig )





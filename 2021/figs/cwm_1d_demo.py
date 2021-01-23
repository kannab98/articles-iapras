from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np


rc.surface.x = [-10, 10]
rc.surface.y = [0, 2]
rc.surface.gridSize = [128, 1]
rc.wind.speed = 7

srf1 = kernel.simple_launch(cuda.cwm)
srf0 = kernel.simple_launch(cuda.default)

srf1[1] = np.rad2deg(np.arctan(srf1[1]))
srf1[2] = np.rad2deg(np.arctan(srf1[2]))
srf0[1] = np.rad2deg(np.arctan(srf0[1]))
srf0[2] = np.rad2deg(np.arctan(srf0[2]))

x = surface.gridx

figs = []
axs = []
lbly = ["$\\zeta(x)$, m", "$\\sigma_x(x)$, deg", "$\\sigma_y(x)$, deg",
         "$v_z(x)$, m/s", "$v_x(x)$, m/s", "$v_y(x)$, m/s"]

nms = ["h", "sigmax", "sigmay", "vz", "vx", "vy"]

for i in range(len(nms)):
    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    figs.append(fig)
    axs.append(ax)
    axs[-1].plot(x, srf0[i])
    axs[-1].plot(x, srf1[i])
    axs[-1].set_xlabel("$x$, m")
    axs[-1].set_ylabel(lbly[i])

    # figs[-1].savefig(nms[i]+'.png')


import matplotlib.backends.backend_pdf
with matplotlib.backends.backend_pdf.PdfPages("cwm_demo_1d_igarss.pdf") as pdf:
    for fig in figs:
        pdf.savefig( fig )





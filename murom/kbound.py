import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from seawave.spectrum import spectrum
from seawave import rc

from cycler import cycler

mpl.style.use(["science"])
# План что я завтра буду писать
## Введение
## Моделирование поверхности
## Выбор граничного волнового числа
## 

waveLength = [0.20,  0.1,  0.05, 0.03, 0.022]
bands = ["L", "S", "C", "X", "Ku"]
rc.antenna.waveLength = waveLength

U10 = np.linspace(3, 15, 128)
X = np.linspace(3000, 20170, 128)

kbU = np.zeros((U10.size, len(waveLength), ))
kbX = np.zeros((X.size, len(waveLength), ))
kbXU = np.zeros((X.size, len(waveLength), ))
peak = np.zeros(X.size)

varU = np.zeros((U10.size, len(waveLength), ))
varX = np.zeros((X.size, len(waveLength), ))

sigmaU = np.zeros((U10.size, len(waveLength), ))
sigmaX = np.zeros((X.size, len(waveLength), ))


for i in range(U10.size):
    rc.wind.speed = U10[i]
    kbU[i] = spectrum.kEdges(waveLength, radar_dispatcher=False)[1:]
    for j in range(len(waveLength)):
        varU[i,j] = spectrum.quad(0,0, k0=0, k1=kbU[i,j])
        sigmaU[i,j] = spectrum.quad(2,0, k0=0, k1=kbU[i,j])

for i in range(U10.size):
    rc.wind.speed = U10[i]
    rc.surface.nonDimWindFetch = X[i]
    kbXU[i] = spectrum.kEdges(waveLength, radar_dispatcher=False)[1:]
    peak[i] = spectrum.peak

rc.wind.speed = 7
for i in range(X.size):
    rc.surface.nonDimWindFetch = X[i]
    kbX[i] = spectrum.kEdges(waveLength, radar_dispatcher=False)[1:]
    for j in range(len(waveLength)):
        varX[i,j] = spectrum.quad(0, 0, k0=0, k1=kbX[i,j])
        sigmaX[i,j] = spectrum.quad(2, 0, k0=0, k1=kbX[i,j])


N = 8
fig = [ plt.figure() for i in range(N) ]
ax  = [ fig[i].add_subplot() for i in range(N) ]

ax[0].plot(U10, kbU)
ax[0].set_xlabel("$U_{10},$ м/с")
ax[0].set_ylabel("$k_b,$ рад/м")

ax[1].plot(peak, kbXU)
ax[1].set_xlabel("$k_m,$ рад/м")
ax[1].set_ylabel("$k_b,$ рад/м")

ax[2].plot(2*np.pi/peak, kbXU)
ax[2].set_xlabel("$\\lambda_m,$ м")
ax[2].set_ylabel("$k_b,$ рад/м")


ax[3].plot(X, kbX)
ax[3].set_xlabel("$\\tilde x$")
ax[3].set_ylabel("$k_b,$ рад/м")


ax[4].plot(U10, varU)
ax[4].set_xlabel("$U_{10},$ м/с")
ax[4].set_ylabel("$\\sigma^2_{\\zeta},\\text{ м}^2$ ")

ax[5].plot(X, varX)
ax[5].set_xlabel("$\\tilde x$")
ax[5].set_ylabel("$\\sigma^2_{\\zeta},\\text{ м}^2$ ")


ax[6].plot(U10, sigmaU)
ax[6].set_xlabel("$U_{10},$ м/с")
ax[6].set_ylabel("$\\sigma^2_{\\xi_{xx}} + \\sigma^2_{\\xi_{yy}}$ ")

ax[7].plot(X, sigmaX)
ax[7].set_xlabel("$\\tilde x$")
ax[7].set_ylabel("$\\sigma^2_{\\xi_{xx}} + \\sigma^2_{\\xi_{yy}}$ ")




from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('spectrum_bounds.pdf') as pdf:
    for i in range(N):
        # Save the subplot.
        ax[i].legend(bands)
        pdf.savefig(figure=fig[i])

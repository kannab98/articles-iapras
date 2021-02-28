import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seawave.spectrum import spectrum
from seawave import rc

waveLength = [0.20,  0.1,  0.05, 0.022]
rc.antenna.waveLength = waveLength

U10 = np.linspace(4, 15, 15)
X = np.linspace(4000, 20170, 15)

kb1 = np.zeros((U10.size, len(waveLength), ))
kb2 = np.zeros((X.size, len(waveLength), ))
kb3 = np.zeros((X.size, len(waveLength), ))

for i in range(U10.size):
    rc.wind.speed = U10[i]
    kb1[i] = spectrum.kEdges(waveLength, radar_dispatcher=False)[1:]

rc.wind.speed = 5
for i in range(X.size):
    rc.surface.nonDimWindFetch = X[i]
    kb2[i] = spectrum.kEdges(waveLength, radar_dispatcher=False)[1:]

rc.wind.speed = 15
for i in range(X.size):
    rc.surface.nonDimWindFetch = X[i]
    kb3[i] = spectrum.kEdges(waveLength, radar_dispatcher=False)[1:]

fig, ax = plt.subplots(ncols=3, figsize=(3*4, 4))
ax[0].plot(U10, kb1)
ax[1].plot(X, kb2)
ax[2].plot(X, kb3)
plt.savefig("kek.png")
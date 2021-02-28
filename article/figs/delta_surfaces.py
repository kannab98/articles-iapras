
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from modeling import cuda, kernel, rc, spectrum, surface

sns.set_style("white")

# Ku, C, S, L-bands
bands = ["L", "S", "C", "Ku"]

rc.wind.speed = 5
rc.surface.gridSize = [1024, 1]
rc.surface.phiSize = 128
rc.surface.kSize = 1024
rc.surface.x = [-5, 5]
rc.surface.y = [-5, 5]

rc.antenna.waveLength = [0.20,  0.1,  0.05, 0.022]
x, y = surface.meshgrid

srf = kernel.launch(cuda.default)
# bands = ['X', 'Ku', 'Ka']
# bands = ["Ku", "C", "S", "L"]

for i in range(len(rc.antenna.waveLength), 0, -1):
    arr = kernel.convert_to_band(srf, i)
    plt.plot(x, arr['elev'], label=bands[i-1])
plt.legend()
plt.savefig('compare_heights.png')

plt.figure()
for i in range(len(rc.antenna.waveLength), 0, -1):
    arr0 = kernel.convert_to_band(srf, 1)
    arr = kernel.convert_to_band(srf, i)
    plt.plot(x, arr0['elev']  - arr['elev'], label=bands[i-1])
plt.legend()
plt.savefig('delta_heights.png')

plt.figure()
for i in range(len(rc.antenna.waveLength), 0,-1):
    arr = kernel.convert_to_band(srf, i)
    plt.plot(x, arr['sx'], label=bands[i-1])
plt.legend()
plt.savefig('compare_slopesxx.png')

plt.figure()
for i in range(len(rc.antenna.waveLength), 0, -1):
    arr0 = kernel.convert_to_band(srf, 1)
    arr = kernel.convert_to_band(srf, i)
    plt.plot(x, arr0['sx']  - arr['sx'], label=bands[i-1])
plt.legend()
plt.savefig('delta_slopesxx.png')
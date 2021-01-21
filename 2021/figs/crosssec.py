from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling import Experiment
sat = Experiment.experiment()


rc.surface.x = [-2500, 2500]
rc.surface.y = [-2500, 2500]
rc.surface.gridSize = [256, 256]
rc.surface.phiSize = 128
rc.surface.kSize = 1024

srf0 = kernel.simple_launch(cuda.default)
srf1 = kernel.simple_launch(cuda.cwm)

X, Y = surface.meshgrid
x = X.flatten()
y = Y.flatten()

print(x[x.size//2])

df = pd.DataFrame({"x": x, "y": y, 
                    "elevation": srf0[0].flatten(),
                    "slopes x":  srf0[1].flatten(),
                    "slopes y":  srf0[2].flatten(),
                    "velocity z": srf0[3].flatten(),
                    "velocity x": srf0[4].flatten(),
                    "velocity y": srf0[5].flatten(), })

df1 = pd.DataFrame({"x": x, "y": y, 
                    "elevation": srf1[0].flatten(),
                    "slopes x":  srf1[1].flatten(),
                    "slopes y":  srf1[2].flatten(),
                    "velocity z": srf1[3].flatten(),
                    "velocity x": srf1[4].flatten(),
                    "velocity y": srf1[5].flatten(), })

surface.coordinates = [x, y, df['elevation'] ]
surface.normal = [df['slopes x'], df['slopes y'], np.ones(x.size) ]
cov = np.cov(df['slopes x'], df['slopes y'])


z = df['elevation'].values.reshape(rc.surface.gridSize)
fig, ax = plt.subplots()
Xi = np.linspace(-14, 14, 50)
N = np.zeros_like(Xi, dtype=float)
for i, xi in enumerate(Xi):
    theta0 = sat.localIncidence(xi=xi)
    ind = sat.sort(theta0, xi=xi)
    theta1 = theta0.reshape(rc.surface.gridSize)
    N[i] = ind[0].size



sigma = surface.cross_section(np.deg2rad(Xi), cov)

A = sigma.max()

ax.plot(Xi, 10*np.log10(A*N/N.max()), label="default")
ax.plot(Xi, 10*np.log10(sigma), label="Теория")
ax.set_xlim([-14, 14])
ax.set_xlabel('$\\theta,$ град')
ax.set_ylabel('$\\sigma_0,$ dB')


surface.coordinates = [x, y, df1['elevation'] ]
surface.normal = [df1['slopes x'], df1['slopes y'], np.ones(x.size) ]
cov = np.cov(df1['slopes x'], df1['slopes y'])


z = df1['elevation'].values.reshape(rc.surface.gridSize)
N = np.zeros_like(Xi, dtype=float)

for i, xi in enumerate(Xi):
    theta0 = sat.localIncidence(xi=xi)
    ind = sat.sort(theta0, xi=xi)
    theta1 = theta0.reshape(rc.surface.gridSize)
    N[i] = ind[0].size

ax.plot(Xi, A*N/N.max(), label="CWM")

df = pd.read_csv('Sig01.txt', sep='\s+', header=None)
sigma = df.values.T
theta = np.linspace(-18, 18, df.shape[1])
ax.plot(theta, np.mean(sigma, axis=1), label="данные DPR")


fig.savefig("crosssec.png")






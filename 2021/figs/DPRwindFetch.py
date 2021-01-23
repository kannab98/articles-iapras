from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling import Experiment
sat = Experiment.experiment()


rc.surface.gridSize = [256, 256]
rc.surface.phiSize = 128
rc.surface.kSize = 1024

rc.wind.speed = 7


U = rc.wind.speed
g = rc.constants.gravityAcceleration
z = rc.antenna.z

xmin = (U**2 * 5000) / g 
xmax = (U**2 * 20170) / g 

x = np.arange(xmin, xmax+1.5e4, 5e3)

fetch = x*g/U**2
fetch[np.where(fetch> 20170)] = 20170
print(fetch)

rc.surface.x = [-2500, 2500]
rc.surface.y = [-2500, 2500]
Xi = np.linspace(-18, 18, 50)

N = np.zeros((fetch.size, Xi.size), dtype=float)

X, Y = surface.meshgrid
for j in range(fetch.size):
    rc.surface.nonDimWindFetch = fetch[j]
    srf0 = kernel.simple_launch(cuda.default)

    df = pd.DataFrame({"x": X.flatten(), "y": Y.flatten(), 
                        "elevation": srf0[0].flatten(),
                        "slopes x":  srf0[1].flatten(),
                        "slopes y":  srf0[2].flatten(),
                        "velocity z": srf0[3].flatten(),
                        "velocity x": srf0[4].flatten(),
                        "velocity y": srf0[5].flatten(), })

    surface.coordinates = [df["x"], df["y"], df['elevation'] ]
    surface.normal = [df['slopes x'], df['slopes y'], np.ones(X.size) ]
    for i, xi in enumerate(Xi):
        theta0 = sat.localIncidence(xi=xi)
        ind = sat.sort(theta0, xi=xi)
        theta1 = theta0.reshape(rc.surface.gridSize)
        N[j][i] = ind[0].size


import pandas as pd
x,y = np.meshgrid(fetch, Xi)
print(x.size, N.size)
df = pd.DataFrame({'fetch':x.flatten(), 'xi':y.flatten(), 'N':N.T.flatten()})
df.to_csv('fetch.tsv' , sep='\t', float_format='%.6E', index=False)

print(N)
plt.contourf(N.T, levels=100)
plt.savefig("fetch_crosssec.png")





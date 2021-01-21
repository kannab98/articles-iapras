from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling import Experiment
sat = Experiment.experiment()


rc.surface.x = [-5, 5]
rc.surface.y = [-5, 5]
rc.wind.speed = 10
rc.surface.gridSize = [256, 256]
rc.surface.phiSize = 128
rc.surface.kSize = 1024

srf = kernel.launch(cuda.default)
srf0 = kernel.convert_to_band(srf, 'Ku')



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

z = df['elevation'].values.reshape(rc.surface.gridSize)


surface.coordinates = [x, y, df['elevation'] ]
surface.normal = [df['slopes x'], df['slopes y'], np.ones(x.size) ]
cov = np.cov(df['slopes x'], df['slopes y'])

theta0 = sat.localIncidence(xi=0)
indKu = sat.sort(theta0)


sat = Experiment.experiment()
srf1 = kernel.convert_to_band(srf, 'Ka')
print(srf1.all() == srf0.all())
df = pd.DataFrame({"x": x, "y": y, 
                    "elevation": srf1[0].flatten(),
                    "slopes x":  srf1[1].flatten(),
                    "slopes y":  srf1[2].flatten(),
                    "velocity z": srf1[3].flatten(),
                    "velocity x": srf1[4].flatten(),
                    "velocity y": srf1[5].flatten(), })

surface.coordinates = [x, y, df['elevation'] ]
surface.normal = [df['slopes x'], df['slopes y'], np.ones(x.size) ]
cov = np.cov(df['slopes x'], df['slopes y'])

theta0 = sat.localIncidence(xi=0)
indKa = sat.sort(theta0)
# theta0 = theta0.reshape(rc.surface.gridSize)

# fig, ax  = plt.subplots(figsize=(5,4))
# X, Y = surface.meshgrid
# mappable = ax.contourf(X,Y,np.rad2deg(theta0))
# fig.colorbar(mappable=mappable)
# fig.savefig('mirrordots.png')


fig, ax  = plt.subplots(figsize=(5,4))
X, Y = surface.meshgrid
mappable = ax.contourf(X,Y, z, levels=100)
ax.scatter(X.flatten()[indKu],Y.flatten()[indKu], marker='.', label="Ku, n=%s " % indKu[0].size)
ax.scatter(X.flatten()[indKa],Y.flatten()[indKa], marker='.', label="Ka, n=%s" % indKa[0].size)
ax.set_xlabel("$X$, м")
ax.set_ylabel("$Y$, м")

bar = fig.colorbar(mappable=mappable)
bar.set_label("$\\zeta$, м")

fig.legend(loc="upper center")
fig.savefig('mirrordots.png')



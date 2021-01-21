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

srf = kernel.launch(cuda.default)
srf0 = kernel.convert_to_band(srf, 'Ku')
# srf1 = kernel.convert_to_band(srf, 'Ka')



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



surface.coordinates = [x, y, df['elevation'] ]
surface.normal = [df['slopes x'], df['slopes y'], np.ones(x.size) ]
cov = np.cov(df['slopes x'], df['slopes y'])


z = df['elevation'].values.reshape(rc.surface.gridSize)

theta0 = sat.localIncidence(xi=0)



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

direction = np.arange(0, 180, 5)
# direction = np.arange(0, 180, 5)
ymax = z*np.arctan(np.deg2rad(18))

rc.surface.x = [-2500, 2500]
rc.surface.y = [-2500, 2500]
Xi = np.linspace(-18, 18, 50)




df = pd.read_csv('direction.tsv', sep="\t", header=0)
N = df["N"]

x, y = np.meshgrid(direction, Xi)

# print(N)
plt.contourf(x,y, N.values.reshape(x.shape), levels=100)
plt.savefig("wind_crosssec1.png")

# # with open("direction.tsv", "r") as f:
# #     df = pd.read_csv(f, sep="\t", header=0)

# # plt.plot(U**2 * df["fetch"]/g, df["sigmaxx"])
# # plt.plot(U**2 * df["fetch"]/g, df["sigmayy"])
# # plt.plot(U**2 * df["fetch"]/g, df["sigmayy"] + df["sigmaxx"])
# # plt.show()


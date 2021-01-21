import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('Sig01.txt', sep='\s+', header=None)

fig, ax = plt.subplots(ncols = 2, figsize=(8,4))
sigma = df.values.T

theta = np.linspace(-18, 18, df.shape[1])
x = np.arange(0, 5e3*df.shape[0], 5e3)

X,Y = np.meshgrid(x, theta)
mappable = ax[0].contourf(X,Y, sigma)

ax[0].set_xlabel("$X,$ м")
ax[0].set_ylabel("$\\theta,$ град")
bar = fig.colorbar(mappable=mappable)
bar.set_label("$\\sigma_0$, dB")

ax[1].plot(theta, sigma[:,0])

fig.savefig('real_crosssec.png')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from modeling import spectrum, surface, rc, kernel, cuda
from scipy.optimize import curve_fit
import pickle as pkl

df = pd.read_csv('Sig01.txt', sep='\s+', header=None)


sigma = df.values.T

theta = np.deg2rad(np.linspace(-18, 18, df.shape[1]))
x = np.arange(0, 5e3*df.shape[0], 5e3)


theta = theta[sigma.shape[0]//2:]
sigma = sigma[sigma.shape[0]//2:]


sigma0 = np.mean(sigma[:,0:50].T, axis=0)
line =  lambda theta, xx, A: 10*np.log10(surface.cross_section(theta,np.array([[xx,0],[0,xx]]))[0])

# Аппроксимация
cond = (np.deg2rad(4) < (theta)) & ((theta) < np.deg2rad(5))
print(cond)
popt = curve_fit(line, 
                    xdata=theta[np.where(cond)],
                    ydata=sigma0[np.where(cond)],
                    p0=[1, 1],
                    bounds=( (0, 0), (np.inf, np.inf) )
                )[0]

A = sigma0.max()
fig, ax  = plt.subplots(ncols = 1, figsize=(4,4))
ax.scatter(np.rad2deg(theta), A*sigma0/sigma0.max(), marker='x', label="DPR data")



# print(popt)
# sigmaapprox = line(theta, *popt)
# ax.plot(np.rad2deg(theta), sigmaapprox/sigmaapprox.max())

cov = np.array([
    [popt[0], 0], 
    [0, popt[0]]
])
print(cov)
s = 10*np.log10(popt[1]*surface.cross_section(theta, cov))
ax.plot(np.rad2deg(theta), A*s/s.max(), color='black', label='theory')

# cond = nnp.abs(theta)<np.deg2rad(4)
# popt = curve_fit(line, 
#                     xdata=theta[np.where(cond)],
#                     ydata=sigma0[np.where(cond)],
#                     p0=[1, 1],
#                     bounds=( (0, 0), (np.inf, np.inf) )
#                 )[0]

# cov = np.array([
#     [popt[0], 0], 
#     [0, popt[0]]
# ])
# print(cov)
# s = 10*np.log10(popt[1]*surface.cross_section(theta, cov))
# ax.plot(np.rad2deg(theta), s)

rc.wind.speed = 2.5
print(spectrum.quad(2,0))
from modeling import Experiment
sat = Experiment.experiment()



rc.surface.x = [-2500, 2500]
rc.surface.y = [-2500, 2500]
rc.surface.gridSize = [512, 512]
rc.surface.phiSize = 128
rc.surface.kSize = 1024

srf0 = kernel.simple_launch(cuda.default)

X, Y = surface.meshgrid
x = X.flatten()
y = Y.flatten()

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
Xi = np.linspace(0, 18, 25)
N = np.zeros_like(Xi, dtype=float)

N = sat.crossSection(Xi)
sdefault = 10*np.log10(N)


plt.scatter(Xi, A*sdefault/sdefault.max(),facecolors='none', edgecolors='r', label="gauss model")
# ax.plot(Xi, sdefault/sdefault.max(), label="default")


# rc.wind.speed = 3.0
# print(spectrum.quad(2,0))
# from modeling import Experiment
# sat = Experiment.experiment()



# rc.surface.x = [-2500, 2500]
# rc.surface.y = [-2500, 2500]
# rc.surface.gridSize = [256, 256]
# rc.surface.phiSize = 128
# rc.surface.kSize = 1024

srf0 = kernel.simple_launch(cuda.cwm)

# X, Y = surface.meshgrid
# x = X.flatten()
# y = Y.flatten()

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

N = sat.crossSection(Xi)
sdefault = 10*np.log10(N)
ax.scatter(Xi, A*sdefault/sdefault.max(), label="choppy model", marker='^')
# srf0 = kernel.simple_launch(cuda.cwm)

# X, Y = surface.meshgrid
# x = X.flatten()
# y = Y.flatten()

# df = pd.DataFrame({"x": x, "y": y, 
#                     "elevation": srf0[0].flatten(),
#                     "slopes x":  srf0[1].flatten(),
#                     "slopes y":  srf0[2].flatten(),
#                     "velocity z": srf0[3].flatten(),
#                     "velocity x": srf0[4].flatten(),
#                     "velocity y": srf0[5].flatten(), })

# surface.coordinates = [x, y, df['elevation'] ]
# surface.normal = [df['slopes x'], df['slopes y'], np.ones(x.size) ]
# cov = np.cov(df['slopes x'], df['slopes y'])

# z = df['elevation'].values.reshape(rc.surface.gridSize)
# Xi = np.linspace(0, 18, 25)
# N = np.zeros_like(Xi, dtype=float)

# N = sat.crossSection(Xi)
# sdefault = 10*np.log10(N)
# ax.plot(Xi, sdefault/sdefault.max(), label="cwm")




# fig, ax = plt.subplots(ncols = 1, figsize=(8,4))
# X,Y = np.meshgrid(x, theta)
# mappable = ax.contourf(X,Y, sigma, levels=100, cmap=cm.plasma)
# ax.set_xlabel("$X,$ м")
# ax.set_ylabel("$\\theta,$ град")
# bar = fig.colorbar(mappable=mappable)
# bar.set_label("$\\sigma_0$, dB")

# ax.plot(theta, sigma[:,0])

ax.legend()
ax.set_xlabel("$\\theta,$ deg")
ax.set_ylabel("$\\sigma_0,$ dB")

with open('crosssec_real.fig', 'wb') as f:
    pkl.dump(fig, f)
# fig.show()
fig.savefig('real_crosssec.png')
import matplotlib.pyplot as plt
import numpy as np
import math as m
import pandas as pd
from matplotlib import cm
from modeling import surface
import pickle as pkl
from scipy.optimize import curve_fit

with open("direction.tsv", 'r') as f:
    df = pd.read_csv(f, sep='\t', header=0)

x = df['direction'].values
y = df['xi'].values
z = 10*np.log10(df['N'].values)

x = x.reshape( m.floor(x.size/36), 36 ) 
y = y.reshape( m.floor(x.size/36), 36 )
z = z.reshape( m.floor(x.size/36), 36 )


fig, ax = plt.subplots(figsize=(4.6, 3.6))
mappable = ax.contourf(x,y,z, levels=100)
bar = fig.colorbar(mappable)
bar.set_label("$\\sigma_0,$ dB")
ax.set_xlabel("$\\phi_0,$ deg")
ax.set_ylabel("$\\theta,$ deg")

fig.savefig('direction.png')
fig, ax = plt.subplots(figsize=(3.6,3.6))

argdir = [0, z.shape[1]//2, z.shape[1]//4, z.shape[1]//8]
argdir = np.sort(argdir)
# argdir = [-1]
direction = np.arange(0, 180, 5) 
xi = np.linspace(-18, 18, 50)

line =  lambda theta, xx, A: 10*np.log10(surface.cross_section(theta,np.array([[xx,0],[0,xx]]))[0])
line0 =  lambda theta, xx, A: 10*np.log10(surface.cross_section(theta,np.array([[xx,0],[0,xx]])))
for i in argdir:
    ax.scatter(y[:,i], z[:,i], marker='x', label="$\\phi_0 = %.0f^\\circ$" % direction[i])
ax.legend()
ax.set_xlabel("$\\theta$, deg" )
ax.set_ylabel("$\\sigma_0$, dB" )


    # popt = curve_fit(line, 
    #                     xdata=np.deg2rad(y[:,i]),
    #                     ydata=z[:,i],
    #                     # p0=[1, 1],
    #                     bounds=( (0, 0), (np.inf, np.inf) )
    #                 )[0]
    # ax.plot(xi, line0(np.deg2rad(xi), *popt))

fig.savefig('direction_slices.png')

with open("fetch.tsv", 'r') as f:
    df = pd.read_csv(f, sep='\t', header=0)

x = df['fetch'].values
y = df['xi'].values

z = 10*np.log10(df['N'].values)

x = x.reshape( m.floor(z.size/19), 19)
x = x[:,:-2]
y = y.reshape( m.floor(z.size/19), 19)
y = y[:,:-2]
z = z.reshape( m.floor(z.size/19), 19)
z = z[:,:-2]


fig, ax = plt.subplots(figsize=(4.6, 3.6))
mappable = ax.contourf(x,y,z, levels=100)
bar = fig.colorbar(mappable)
bar.set_label("$\\sigma_0,$ dB")
ax.set_xlabel("$\\tilde x$")
ax.set_ylabel("$\\theta,$ deg")

fig.savefig('fetch.png')

fig, ax = plt.subplots(figsize=(3.6,3.6))

argxi = [-1, -z.shape[0]//2, -z.shape[0]//4, -z.shape[0]//8]
argxi = np.sort(argxi)
for i in argxi:
    ax.scatter(x[i,:], z[i,:], marker='x')
    f = np.polyfit(x[i,:], z[i,:], deg=1)
    f = np.poly1d(f)
    ax.plot(x[i,:], f(x[i,:]), 
        label="$\\theta=%.0f^\\circ$" % xi[i])

ax.set_xlabel('$\\tilde x$')
ax.set_ylabel('$\\sigma_0,$ dB')

ax.legend()
fig.savefig('fetch_slices.png')
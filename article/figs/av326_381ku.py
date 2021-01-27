import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os, re
import pickle as pkl

from scipy.optimize import curve_fit

# regex = re.compile('(.*db.dat)')
regex = re.compile('(.*dat)')

def find_sigma(x, y, in_type=None, out_type=None):
    if in_type == "dB":
        y = 10**(y/10)
    
    popt = curve_fit(cross_section, 
                        xdata=np.deg2rad(df[0].values[cond]),
                        ydata=df[1].values[cond],
                        p0 = [1, 1],
                        bounds=( (1e-4, 1e-4), (np.inf, np.inf) )
                    )[0]

    return popt


def cross_section(theta, xx, F, out_type=None): 
    # F = 1
    F = 0.8
    sigma =  F**2/( 2*np.cos(theta)**4 * np.sqrt(xx*xx) )
    sigma *= np.exp( - np.tan(theta)**2 / (2*xx*xx) )

    if out_type == "dB":
        sigma = 10*np.log10(sigma)

    return sigma

fig, ax = plt.subplots()

for root, dirs, files in os.walk(os.getcwd()):
  for file in files:
    if regex.match(file):
        df = pd.read_csv(file, sep="\s+", header=None)
        cond = np.where(df[0].values > 4)

        if file.split('.')[0][-2:] == 'db':
            func = lambda x: 10**(x/10)
        else:
            func = lambda x: x


        popt = find_sigma(np.deg2rad(df[0].values[cond]), func(df[1].values[cond]))

        y = func(df[1])
        ax.plot(df[0], y/y.max())

        fn = file.split(".")[0]
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\\sigma_0$")

        y = cross_section(np.deg2rad(df[0].values), *popt)
        ax.plot(df[0], y/y.max())
        print(popt)

fig.savefig(fn)
with open(fn+'.fig', 'wb') as f:
    pkl.dump(fig, f)



# files = ["k"]

# with open("")
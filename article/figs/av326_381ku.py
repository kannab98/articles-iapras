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
    
    y = np.log(y*np.cos(x)**4)
    x = np.tan(x)**2
    popt = np.polyfit(x, y, deg=1)
    popt[1] = np.exp(popt[1])
    popt[0] = -1/2/popt[0]
    return popt


# def cross_section(theta, xx, F, out_type=None): 
#     sigma =  F**2/( 2*np.cos(theta)**4 * np.sqrt(xx*xx) )
#     sigma *= np.exp( - np.tan(theta)**2 / (2*xx*xx) )

#     if out_type == "dB":
#         sigma = 10*np.log10(sigma)

#     return sigma

def cross_section(theta, xx, sigma0, out_type=None): 
    sigma = sigma0/np.cos(theta)**4
    sigma *= np.exp( - np.tan(theta)**2 / (2*xx) )

    if out_type == "dB":
        sigma = 10*np.log10(sigma)

    return sigma

fig, ax = plt.subplots()

for root, dirs, files in os.walk(os.getcwd()):
  for file in files:
    if regex.match(file):
        df = pd.read_csv(file, sep="\s+", header=None)
        cond = np.where((df[0].values > 4) & (df[0].values < 10))

        if file.split('.')[0][-2:] == 'db':
            func = lambda x: 10**(x/10)
        else:
            func = lambda x: x


        popt = find_sigma(np.deg2rad(df[0].values[cond]), func(df[1].values[cond]))

        y = func(df[1])
        ax.plot(df[0], y)

        fn = file.split(".")[0]
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\\sigma_0$")

        y = cross_section(np.deg2rad(df[0].values), *popt)
        ax.plot(df[0], y)
        print(popt)

fig.savefig(fn + '.png')
with open(fn+'.fig', 'wb') as f:
    pkl.dump(fig, f)



# files = ["k"]

# with open("")
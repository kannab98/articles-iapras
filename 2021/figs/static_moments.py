from modeling import rc, spectrum
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

U = np.linspace(3,15)
sigma = np.zeros_like(U)
mean = np.zeros_like(U)
var = np.zeros_like(U)
sigmax = np.zeros_like(U)
sigmay = np.zeros_like(U)


for i in range(U.size):
    rc.wind.speed = U[i]
    var[i] = spectrum.quad(0,0)
    mean[i] = -spectrum.quad(1,0)
    print(mean[i])
    # sigmax[i] = spectrum.dblquad(2,0,0)
    # sigmay[i] = spectrum.dblquad(0,2,0)


figs = []
axs = []

fig, ax = plt.subplots()


ax.plot(U, var, label="Gaussian")
ax.plot(U, var - mean**2, label="CWM")
ax.legend()
ax.set_xlabel("$U_{10}$, м/с")
ax.set_ylabel("$\\sigma^2_0, \\text{м}^2$")

fig.savefig("cwm_variance")
df = pd.DataFrame({"U": U, "mean": mean, "var":var, "sigmaxx":sigmax, "sigmayy":sigmay})
df.to_csv('cwm_first_moments.tsv', sep="\t", index=False, float_format="%.8E")
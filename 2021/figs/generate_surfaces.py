import pandas as pd
import numpy as np
from modeling import rc 
from modeling import surface, kernel, cuda
from modeling import spectrum 
import matplotlib.pyplot as plt
from scipy.integrate import quad

U = np.linspace(5, 15, 5)
X,Y = surface.meshgrid
mean = np.zeros_like(U)
var = np.zeros_like(U)
var0 = np.zeros_like(U)
var1 = np.zeros_like(U)

mean0 = np.zeros_like(U)
mean1 = np.zeros_like(U)

for i in range(U.size):
    rc.wind.speed = U[i]
    var[i] = spectrum.quad(0,0)
    mean[i] = -spectrum.quad(1,0)

    arr = kernel.launch(cuda.default)
    arr = kernel.convert_to_band(arr, 'Ka')
    # arrKu = kernel.convert_to_band(arr, 'Ku')

    arr0 = kernel.launch(cuda.cwm)
    arr0 = kernel.convert_to_band(arr0, 'Ka')
    print(arr[0][0], arr0[0][0])
    # arr0Ku = kernel.convert_to_band(arr, 'Ku')

    moments = surface.staticMoments(X,Y, arr)
    moments0 = surface.staticMoments(X,Y, arr0)

    mean0[i] = moments0[0]
    mean1[i] = moments[0]

    var0[i] = moments0[1]
    var1[i] = moments[1]

ax.plot(U, var, label="Gaussian")
ax.plot(U, var - mean, label="CWM")
ax.plot(U, var0, label="CWM_real")
ax.plot(U, var1, label="Gaussian_real")
ax.legend()
ax.set_xlabel("$U_{10}$, м/с")
ax.set_ylabel("$\\sigma^2_0, \\text{м}^2$")
fig.savefig("variance_real")

# with open("data/ModelMoments.xlsx", "wb") as f:
            # df = df.to_excel(f)


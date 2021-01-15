import numpy as np
from numpy import exp
import matplotlib.pyplot as plt 

def dispersion(k):
    k = abs(k)
    return np.sqrt(9.81*k + 74e-6*k**3)


A = 1

k = 1

x = np.linspace(-2*np.pi, 2*np.pi, 500)
x0 = np.linspace(-2*np.pi, 2*np.pi, 500)
t = 0

kr = k*x*exp(1j*0)
w = dispersion(k)
e = A * exp(1j*kr)  * exp(1j*w*t)

surface = np.zeros((5, x.size))
# Высоты (z)
surface[0,:] =  +e.real
# Наклоны X (dz/dx)
surface[1,:] =  -e.imag * k.real
# Наклоны Y (dz/dy)
surface[2,:] =  -e.imag * k.imag
# Орбитальные скорости Vz (dz/dt)
surface[3,:] =  -e.imag * w

# Vh -- скорость частицы вдоль направления распространения ветра.
# см. [shuleykin], гл. 3, пар. 5 Энергия волн.
# Из ЗСЭ V_h^2 + V_z^2 = const

# Орбитальные скорости Vh
surface[4,:] += e.real * w 


fig1, ax1 = plt.subplots(ncols=1, nrows=1)
# ax1.set_title("Высоты")
ax1.plot(x, surface[0,:])
ax1.set_ylabel("$\\zeta(x)$, м")
ax1.set_xlabel("$x$, м")

fig2, ax2 = plt.subplots(ncols=1, nrows=1)
# ax2.set_title("Наклоны")
ax2.plot(x, np.rad2deg(np.arctan(surface[1,:])))
ax2.set_ylabel("$\\sigma(x)$, град")
ax2.set_xlabel("$x$, м")

fig3, ax3 = plt.subplots(ncols=1, nrows=1)
# ax3.set_title("Скорости Z")
ax3.plot(x, surface[3,:])
ax3.set_ylabel("$v_z(x)$, м/c")
ax3.set_xlabel("$x$, м")

fig4, ax4 = plt.subplots(ncols=1, nrows=1)
# ax4.set_title("Скорости X")
ax4.plot(x, surface[4,:])
ax4.set_ylabel("$v_x(x)$, м/c")
ax4.set_xlabel("$x$, м")

V1 = surface[3,:]**2 + surface[4,:]**2

# Поправка на наклоны заостренной поверхности
# Наклоны X dz/dx * dx/dx0


# ax[0,1].plot(x0, surface[1,:])
surface[1] *= 1 - e.real * (k.real * k.real)/abs(k)
# Наклоны Y dz/dy * dy/dy0 
surface[2] *= 1 - e.real * (k.imag * k.imag)/abs(k)
# Орбитальные скорости Vh dVh/dx * dx/dx0
surface[4] *= 1 - e.real * (k.real * k.real)/abs(k)


x -= e.imag * k.real/abs(k)
ax1.plot(x, surface[0,:])
# x += e.imag * k.real/abs(k)

ax2.plot(x, np.rad2deg(np.arctan(surface[1,:])))
ax3.plot(x, surface[3,:])
ax4.plot(x, surface[4,:])

V2 = surface[3,:]**2 + surface[4,:]**2

fig, ax = plt.subplots()
ax.plot(V1)
ax.plot(V2)

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("cwm_demo.pdf")
for fig in [fig1, fig2, fig3, fig4]:
    pdf.savefig( fig , bbox_inches="tight")

pdf.close()
plt.show()
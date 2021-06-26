import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
from multiprocessing import Pool, cpu_count
from scipy.constants import g, m_n
from scipy.optimize import curve_fit


# Potential
alpha = 10*np.pi/180
F = g * m_n * np.sin(alpha)

# Sim
Nx = 30e4
Sx = 30e-3

# Source
sw = 500e-6
Nn = 1
wvl = 5e-10
theta = 0.2*np.pi/180

Nit = 1
Nn = 1

# Setup
L = 8.8
Ls2 = 4.75
D1 = 1
D = 1e-2 # D3-D1

L1 = Ls2 - D1
D3 = D1 + D
L3 = L - Ls2 - D3

# Gratings
P = 2.4e-6
phi1 = np.pi*0.5
phi2 = np.pi
phi3 = np.pi*0.5

# Plotting
xmin = -20e-3
xmax = 20e-3
numbins = int(4e5)

# MAIN SCRIPT

wave = NeutronWave(Nx, Sx, Nn=Nn, wvl=wvl)
wave.slitSource(L1, sw, theta=0, randPos=False)

print(wave.theta)

wave.rectPhaseGrating(P, phi1)
wave.propagate_linear_potential(D1, F, pad=True)

print(wave.theta)

wave.rectPhaseGrating(P, phi2)
wave.propagate_linear_potential(D3, F, pad=False)

print(wave.theta)

wave.rectPhaseGrating(P, phi3)
wave.propagate_linear_potential(L3, F, pad=False)
#wave.propagate(L3, pad=False)

print(wave.theta)

center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)

print(center[np.argmax(htemp)])

#plt.plot(center, htemp)
plt.plot(wave.X[0], np.abs(wave.Psi[0])**2)
plt.show()

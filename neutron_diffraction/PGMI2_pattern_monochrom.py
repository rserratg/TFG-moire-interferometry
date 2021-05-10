# 2PGMI

# Parameters to compare with results from monochromatic source in 
#   Pushin et. al. (2017), 
#   Far-field interference of a neutron white beam and the applications to 
#   noninvasive phase-contrast imaging

# Output fitted to a sine with fixed frequency

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit

Nx = 5e4
Sx = 10e-3
sw = 200e-6
L1 = 1.2
L = 2.99
D = 12e-3
L2 = L - L1 - D

# Gratings
P = 2.4e-6
phi = 0.27*np.pi # Actually, np.pi * 0.27. Best contrast with 0.5*np.pi (in theory)

print('Starting sim')
wave = NeutronWave(Nx, Sx, Nn=1, wvl=0.44e-9)

print(f'Sampling: {wave.d*1e6} um')

print('Slit source')
wave.slitSource(L1, sw, theta=0.5*np.pi/180)

print('G1 and first propagation')
wave.rectPhaseGrating(P, phi)
wave.propagate(D)

print('G2 and second propagation')
wave.rectPhaseGrating(P, phi)
wave.propagate(L2)

print('Plotting')

datamin = -10e-3
datamax = 10e-3
numbins = int(1e4)
center, myhist = wave.hist_intensity(numbins, xlimits=(datamin,datamax), retcenter=True)

width = center[1]-center[0]
#plt.bar(center*1e3, myhist, align='center', width=width*1e3)
plt.plot(center*1e3, myhist, '-')

C, Pd, fit = contrast_fit(center, myhist, P*L/D)
plt.plot(center*1e3, fit, '--', color='orange')

print('Period:', Pd*1e3, 'mm')
print('Freq:', 1/Pd)

plt.xlabel('x [mm]')
plt.show()

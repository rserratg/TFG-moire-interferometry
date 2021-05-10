# 3PGMI

# Parameters to compare with results in
#   Sarenac et. al. (2018), 
#   Three Phase-Grating Moiré Neutron Interferometer for Large Interferometer Area Applications

# Output fitted to a sine with fixed frequency
# Simulation run multiple times to increase total neutron count

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit

   
# Sim
Nx = 8e4
Sx = 8e-3
Nit = 100

# Source
sw = 500e-6
Nn = 100
wvl = 0.5e-9
theta = 0.2*np.pi/180

# Setup
L = 8.8
Ls2 = 4.75
D1 = 4.6e-2
D = -2.5e-2 # D3 - D1

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
numbins = int(2e4)

center = np.empty(numbins-1)
hist = np.zeros(numbins-1)

for ii in range(Nit):

    print()
    print("Iteration:", ii+1)
    print("  First prop")
    wave = NeutronWave(Nx, Sx, Nn=Nn, wvl=wvl)
    wave.slitSource(L1, sw, theta=theta, randPos=True)
    print("  Second prop")
    wave.rectPhaseGrating(P,phi1)
    wave.propagate_nopad(D1)
    print("  Third prop")
    wave.rectPhaseGrating(P,phi2)
    wave.propagate_nopad(D3)
    print("  Last prop")
    wave.rectPhaseGrating(P,phi3)
    wave.propagate_nopad(L3)
    print("  Results")
    center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
    hist += htemp/Nit
    
print()
print('Period:', abs(P*L/D)*1e3, 'mm')

width = center[1]-center[0]
plt.plot(center*1e3, hist, '-')

C, Pd, fit = contrast_fit(center, hist, abs(P*L/D), fitP=True)
print('Period fit:', Pd)
print('Contrast:', C)
plt.plot(center*1e3, fit, '--', color='orange')

plt.xlabel('x [mm]')
plt.show()
    
    
    

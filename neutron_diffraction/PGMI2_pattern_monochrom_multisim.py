# 2PGMI

# Parameters to compare with results form monochromatic source in
#   Pushin et. al. (2017), 
#   Far-field interference of a neutron white beam and the applications to 
#   noninvasive phase-contrast imaging

# Output fitted to a sine with fixed frequency
# Simulation run multiple times to increase total neutron count

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from scipy.optimize import curve_fit

def fitSin(x, u, P, xlim = None):

    xaux = x
    if xlim is not None:
        xmin, xmax = xlim
        cond1 = x >= xmin
        cond2 = x <= xmax
        cond = cond1 & cond2
        xaux = x[cond]
        u = u[cond]
    
    # Sin function to fit
    def fun(xx, a, b, c):
        return a + b*np.sin(2*np.pi*xx/P + c)
    
    # Fit function to data and retrieve optimal parameters
    popt, _ = curve_fit(fun, xaux, u)
    A, B, phi = popt
    
    # Calculate contrast
    C = abs(B/A)
    fit = fun(x, A, B, phi)
    return C, fit
    
# Sim
Nx = 5e4
Sx = 10e-3

# Source
sw = 200e-6
Nn = 500
wvl = 0.44e-9

# Setup
L1 = 1.2
L = 2.99
D = 12e-3
L2 = L - L1 - D

# Gratings
P = 2.4e-6
phi = 0.27*np.pi

# Plotting
xmin = -10e-3
xmax = 10e-3
numbins = int(1e4)

center = np.empty(numbins-1)
hist = np.zeros(numbins-1)

for ii in range(100):

    print()
    print("Iteration:", ii+1)
    print("  First prop")
    wave = NeutronWave(Nx, Sx, Nn=Nn, wvl=wvl)
    wave.slitSource(L1, sw, theta=0.5*np.pi/180)
    print("  Second prop")
    wave.rectPhaseGrating(P,phi)
    wave.propagate(D)
    print("  Last prop")
    wave.rectPhaseGrating(P,phi)
    wave.propagate(L2)
    print("  Results")
    center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
    hist += htemp/100
    
print()
print('Period:', P*L/D*1e3, 'mm')

width = center[1]-center[0]
plt.plot(center*1e3, hist, '-')

C, fit = fitSin(center, hist, P*L/D)
print('Contrast:', C)
plt.plot(center*1e3, fit, '--', color='orange')

plt.xlabel('x [mm]')
plt.show()
    
    
    

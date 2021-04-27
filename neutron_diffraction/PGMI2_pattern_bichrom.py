# 2PGMI

# Parameters to compare with results from bichromatic source in
#   Pushin et. al. (2017)
#   Far-field interference of a neutron white beam and the applications to
#   noninvasive phase-contrast imaging

# Output fitted to a sine with fixed frequency

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

# PARAMETERS

# Numerical params
Nx = 5e4
Sx = 10e-3

# Setup
sw = 200e-6
L1 = 1.73
L = 3.52
D = 12e-3
L2 = L - L1 - D

# Gratings
P = 2.4e-6
phi = 0.5*np.pi

# wavelength
wvlvals = np.array([0.22e-9, 0.44e-9])
wvlweights = np.array([1/3, 2/3])

# Plotting
xmin = -10e-3
xmax = 10e-3
numbins = int(1e4)

# SIMULATION

center = np.empty(numbins-1)
hist = np.zeros(numbins-1)

for wvl, weight in zip(wvlvals, wvlweights):
    
    print(f"Sim wavelength: {wvl}")
    wave = NeutronWave(Nx, Sx, Nn=500, wvl=wvl)
    print("Slit source")
    wave.slitSource(L1,sw,theta=0.5*np.pi/180)
    print("G1 and first propagation")
    wave.rectPhaseGrating(P,phi)
    wave.propagate(D)
    print("G2 and second propagation")
    wave.rectPhaseGrating(P,phi)
    wave.propagate(L2)
    
    print("Store results")
    center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
    hist += htemp*weight
    
print('Plotting')

width = center[1] - center[0]
plt.plot(center*1e3, hist, '-')

C, fit = fitSin(center, hist, P*L/D)
plt.plot(center*1e3, fit, '--', color='orange')

plt.xlabel('x [mm]')
plt.show()

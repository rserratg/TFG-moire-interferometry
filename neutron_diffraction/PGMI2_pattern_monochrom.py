# 2PGMI

# Parameters to compare with results from monochromatic source in 
#   Pushin et. al. (2017), 
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
wave = NeutronWave(Nx, Sx, Nn=1000, wvl=0.44e-9)

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

C, fit = fitSin(center, myhist, P*L/D)
plt.plot(center*1e3, fit, '--', color='orange')

plt.xlabel('x [mm]')
plt.show()

print('Period:', P*L/D*1e3, 'mm')
print('Freq:', D/(P*L))

ft = np.fft.rfft(myhist)
freq = np.fft.rfftfreq(len(myhist), center[1]-center[0])
plt.plot(freq, np.abs(ft))
plt.show()

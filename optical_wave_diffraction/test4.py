# 2PGMI - Contrast of polychromatic, extended source
# Miao et. al. 2016

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity, contrast_period
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
import json

# Numerical parameters
N = 8e4
Sx = 8e-2

# Beam parameters
wvlvals = np.linspace(450, 700, 25)
wvlweights = 2.5e6*skewnorm.pdf(wvlvals, 5, loc=500, scale=100)
wvlvals *= 1e-9

theta_max = 8*np.pi/180
theta_num = 151

# Propagation
L1 = 10e-2
L = 20e-2
D = 1.1e-3

# Grating settings
P = 14.3e-6
phi = np.pi/2


# AUXILIARY FUNCTIONS

'''
    Contrast by fitting sine
    Return error in contrast
'''
def contrast_error(x, u, P, xlim = None):

    # Get only data inside interval limited by xlim
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
    popt, pcov = curve_fit(fun, xaux, u)
    A, B, phi = popt
    
    # Calculate contrast
    C = abs(B/A)
    
    # Contrast error
    perr = np.sqrt(np.diag(pcov))
    Aerr, Berr, _ = perr
    Cerr = Berr/A + Aerr*B/A**2
    
    fit = fun(x, A, B, phi)
    return C, Cerr, fit


# SIMULATION

# Get x-space (independent of wavelength)
ref = OptWave(N, Sx, wvl=None)
x = ref.x

# Run sim
print('Running simulation...')

I = np.zeros(int(N))

for wvl, weight in zip(wvlvals, wvlweights):

    #print()
    print(f"wavelength: {wvl}")
    #print()

    Iwvl = np.zeros(int(N))

    for theta in np.linspace(0, theta_max, theta_num):
    
        #print(f"theta: {theta}")

        wave = OptWave(N,Sx,wvl)
        wave.planeWave(theta=theta)
        wave.rectAperture(0.44e-3)
        wave._conv_propagation(L1, "AS", pad=False)
        wave.rectPhaseGrating(P, phi)
        wave._conv_propagation(D, "AS", pad=False)
        wave.rectPhaseGrating(P, phi)
        wave._conv_propagation(L-D-L1, "AS", pad=False)
        
        Ith = intensity(wave.U)
        Iwvl += Ith
        
        # Use symmetry to calculate negative angles
        if theta > 0:
            Iwvl[1:] += np.flip(Ith)[:-1]
        
    I += Iwvl * weight
    
Pmoire = P*L/D

_, Pd, _ = contrast_period(x, I, Pmoire, xlim=(-12e-3, 12e-3))
C, Cerr, fit = contrast_error(x, I, Pd, xlim=(-12e-3, 12e-3))
fd = 1/Pd

print()
print(f"Contrast: {C}")
print(f"Contrast error: {Cerr}")
print(f"Period: {1/fd*1e3} mm")

plt.plot(x*1e3, I)
plt.plot(x*1e3, fit, '--')
plt.xlabel('x [mm]')
plt.ylabel('Intensity [a.u.]')
plt.show()

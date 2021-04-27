import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity, contrast_FT, contrast_period, contrast
from scipy.stats import skewnorm

N = 1e5
Sx = 10e-2

wvlvals = np.linspace(450, 700, 25)
wvlweights = 2.5e6*skewnorm.pdf(wvlvals, 5, loc=500, scale=100)
wvlvals *= 1e-9

theta_max = 8*np.pi/180
theta_num = 201

L1 = 10e-2
D = 0.1e-3
L = 20e-2

P = 14.3e-6
phi = np.pi/2

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
    
# Field data
ref = OptWave(N,Sx,None)
x = wave.x
d = wave.d

# Contrast: compare methods
# - Period and contrast with FT
# - Period and contrast with fit
# - Contrast with fit (fixed period - FT)
# - Contrast with fit (fixed period - fit)
# - Contrast with fit (fixed period - theoretical)
Pd0 = P*L/D
C, fd = contrast_FT(d, I, 1/Pd0)

xlim = (-10e-3, 10e-3)
C2, Pd2, _, fit2 = contrast_period(x, I, Pd0, xlim=xlim)
C3, fit3 = contrast(x, I, 1/fd, xlim=xlim, retfit=True)
C4 = contrast(x, I, Pd2, xlim=xlim)
C5 = contrast(x, I, Pd0, xlim=xlim)

print()
print(f"Sampling: {d}")
print(f"Period: {1/fd*1e3} mm")
print(f"Contrast: {C}")
print()
print(f"Period fit: {Pd2*1e3} mm")
print(f"Contrast fit: {C2}")
print()
print("Contrast fit fixed")
print(f"FT period: {C3}")
print(f"Fit period: {C4}")
print(f"Analytical period: {C5}")

# Plot
plt.plot(x*1e3, I)
plt.plot(x*1e3, fit2, '--')
plt.plot(x*1e3, fit3, ':')
plt.xlabel('x [mm]')
plt.show()

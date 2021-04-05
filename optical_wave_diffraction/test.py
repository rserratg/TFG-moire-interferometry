# Test: fit to cos*cos

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity
from scipy.optimize import curve_fit

# A + B*cos(2pi*fp*x + phi1)*cos(2pi*fm*x + phi2)
def contrast_frequency(x, u, fp0, fm0, xlim = None):
    
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
    def fun(xx, a1, b1, c1, d1, a2, b2, c2, d2):
        
        cos1 = 1 + 1*np.cos(2*np.pi*c1*xx + d1)
        cos2 = a2 + b2*np.cos(2*np.pi*c2*xx + d2)
        
        return cos1*cos2
    
    # Fit function to data and retrieve optimal parameters
    p0 = (1, 1, fp0, 0, 1, 1, fm0, 0)
    bmin = [0, 0, 0, -np.pi, 0, 0, 0, -np.pi]
    bmax = [np.inf, np.inf, np.inf, np.pi, np.inf, np.inf, np.inf, np.pi]
    popt, pcov = curve_fit(fun, xaux, u, p0=p0, bounds=(bmin, bmax))
    
    #A, B, fp, fm, phi1, phi2 = popt
    fp = popt[2]
    fm = popt[6]
    
    # Calculate contrast
    C = 0
    
    print(popt)
    
    fit = fun(x, *popt)
    return C, fp, fm, fit
        
       
# Numeric parameters
N = 5e5
S = 60e-3 # field size

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Grating settings
P = 180e-6

# Lens
f = -75e-3

# Propagation
L0 = 11e-2
L1 = 32e-2
D = 6.5e-2
L = 1
L2 = L - (-f + L1 + D)

if L2 < 0:
    print('Invalid parameters: negative propagation')
    exit()      

print('Calculating output pattern...')

# Output pattern
wave = OptWave(N,S,wvl)
wave.gaussianBeam(w0, z0=z0)
wave.angular_spectrum(L0)
wave.lens(f)
wave.angular_spectrum(L1)
wave.rectPhaseGrating(P, np.pi/2)
wave.angular_spectrum(D)
wave.rectPhaseGrating(P, np.pi/2)
wave.angular_spectrum(L2)

print('Calculating reference field...')

# Reference pattern
ref = OptWave(N,S,wvl)
ref.gaussianBeam(w0, z0=z0)
ref.angular_spectrum(L0)
ref.lens(f)
ref.angular_spectrum(L + f)

# Get results
x = wave.x
I = intensity(wave.U)
Iref = intensity(ref.U)
Is = I/Iref

'''
print('Rebinning...')

x, Is = rebin(x, Is, 500, avg=True)
print('Rebin px:', x[1]-x[0])
'''

print('Fitting...')

# Calculate contrast from known period
Pd = P*L/D
c, fp, fm, fit = contrast_frequency(x, Is, 1/0.44*1e3, 1/Pd, xlim=(-15e-3,15e-3))

print('Expected moire period:', Pd*1e3, 'mm')
print('Contrast:', c)
print('Fitted period 1:', 1/fp*1e3, 'mm')
print('Fitted period 2:', 1/fm*1e3, 'mm')

print('Plotting...')

plt.plot(x*1e3, I/Iref)
plt.plot(x*1e3, fit, '--')
plt.xlabel('x [mm]')
plt.ylabel('Intensity [arb. units]')
plt.show()

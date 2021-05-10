# Test - 2PGMI
# Plot simulation of pattern at camera
# Plot profile (pi + 4*cos(2pi*f1*x))**2 * (pi + 4*cos(2pi*f2*x))**2 / (4*pi**4)

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity    
    
###################################

# GENERAL PARAMETERS

numsim = 1

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Grating parameters
P = 180e-6

# Lens
f = -75e-3
    
# Numerical parameters
N = 5e5
S = 100e-3

# Setup
L0 = 11e-2
L1 = 32e-2
D = 2.7e-2
L = 1
L2 = L - (-f + L1 + D)
    
print('Calculating output intensity...')

wave = OptWave(N, S, wvl)
wave.gaussianBeam(w0, z0=z0)
wave.angular_spectrum(L0)
wave.lens(f)
wave.angular_spectrum(L1)
wave.rectPhaseGrating(P, np.pi/2)
wave.angular_spectrum(D)
wave.rectPhaseGrating(P, np.pi/2)
wave.angular_spectrum(L2)

# Get results
x = wave.x
d = wave.d
I = intensity(wave.U)

# Plot output intensity
plt.plot(x*1e3, I, '.')
plt.xlabel('x [mm]')
plt.ylabel('Intensity [a.u.]')


f1 = 2194
f2 = 2344
I2 = (np.pi + 4*np.cos(2*np.pi*f1*x))**2 * (np.pi + 4*np.cos(2*np.pi*f2*x))**2
I2 /= 4*np.pi**4
I2 *= I.max()/(2*np.pi)
plt.plot(x*1e3, I2, '--')

plt.show()

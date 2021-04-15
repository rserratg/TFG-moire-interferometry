# Three phase-grating moire interferometer
# Pattern after system
# Result fitted to sinusoid

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity, contrast, rebin

# Sim parameters
N = 5e5
L = 60e-3

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L0 = 11e-2
L1 = 32e-2
D1 = 10e-2
D3 = 30e-2
L3 = 48e-2

# Grating settings
P = 180e-6
phi1 = np.pi/2
phi2 = np.pi
phi3 = np.pi/2

# Lens
f = -75e-3

print('Calculating field')

wave = OptWave(N,L,wvl)
wave.gaussianBeam(w0, z0=z0)
wave.angular_spectrum(L0)
wave.lens(f)
wave.angular_spectrum(L1)
wave.rectPhaseGrating(P, phi1)
wave.angular_spectrum(D1)
wave.rectPhaseGrating(P, phi2)
wave.angular_spectrum(D3)
wave.rectPhaseGrating(P, phi3)
wave.angular_spectrum(L3)

print('Calculating reference field')

ref = OptWave(N,L,wvl)
ref.gaussianBeam(w0, z0=z0)
ref.angular_spectrum(L0)
ref.lens(f)
ref.angular_spectrum(L1 + D1 + D3 + L3)

x = wave.x
I = intensity(wave.U)
Iref = intensity(ref.U)

print('Fitting')

Lt = -f + L1 + D1 + D3 + L3
Pd = P*Lt/(D3 - D1)

c, fit = contrast(x, I/Iref, Pd, xlim=(-10e-3, 10e-3), retfit = True)

print('Results:')
print('C =', c)
print('Pd =', Pd)
print('Px =', (x[1]-x[0])*1e3)

plt.plot(x*1e3, I/Iref)
plt.plot(x*1e3, fit, '--')
plt.show()

plt.plot(x*1e3, I)
plt.plot(x*1e3, Iref*fit, '--')
plt.show()


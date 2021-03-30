# Test contrast function

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity, contrast, rebin

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L1 = 40e-3
L2 = 150e-3
D = 20e-3
L3 = 50e-3

# Grating settings
P = 180e-6

# Lens
f = -25e-3

# Parameters
N = 1e6
L = 50e-3
phi = np.pi/2

# Field
wave = OptWave(N,L,wvl)
wave.gaussianBeam(w0, z0=z0)  
wave.angular_spectrum(L1)
wave.lens(f)
wave.angular_spectrum(L2)
wave.rectPhaseGrating(P, phi)
wave.angular_spectrum(D)
wave.rectPhaseGrating(P, phi)
wave.angular_spectrum(L3)

# Ref field
ref = OptWave(N,L,wvl)
ref.gaussianBeam(w0, z0=z0)  
ref.angular_spectrum(L1)
ref.lens(f)
ref.angular_spectrum(L2+D+L3)


# Results
x = wave.x
I = intensity(wave.U)
Iref = intensity(ref.U)

# Rebin
#xaux, I = rebin(x, I, 500, avg=False)
#_, Iref = rebin(x, Iref, 500, avg=False)
#x = xaux
#print(x[1]-x[0])

plt.plot(x, I)
plt.plot(x, Iref)
plt.show()


# Expected period of fringes
L = -f + L2 + D + L3
Pd = P*L/D
print("Pd:", Pd*1e3, "mm")

# Contrast
c, fit = contrast(x, I/Iref, Pd, xlim=(-10e-3, 10e-3), retfit = True)
print("Contrast:", c)

# Plot
plt.plot(x*1e3, I/Iref)
plt.plot(x*1e3, fit, '--')
plt.show()

# General example

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalizedIntensity

#Sim parameters
N = 1000# number of grid points
L = 4e-3 # total size of the grid [m]
wvl = 1e-6 # optical wavelength [m]
D = 0.5e-3
z = .1 # propagation distance [m]

# Bandlimit for AS
# w/o zero-padding: 2e4
# w/ zero-padding: ~ 4e4

# Sim computation
wave = OptWave(N,L,wvl)
wave.planeWave()

# Element
#wave.rectAmplitudeGrating(0.1e-3)
#wave.rectPhaseGrating(0.5e-3, np.pi)
#wave.doubleSlit(1e-3, D)
wave.rectAperture(D)

# Propagation
#wave.fraunhofer(z)
wave.fresnel_DI(z)
#wave.fresnel_CV(z)
#wave.fresnel_AS(z)
#wave.rayleigh_sommerfeld(z, fast=False)
#wave.angular_spectrum_repr(z)

# Get results
xout = wave.x
Uout = wave.U
#I = normalizedIntensity(Uout)
#I = np.abs(Uout)
I = np.angle(Uout)

# Plot results
plt.plot(xout, I, "-")
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()

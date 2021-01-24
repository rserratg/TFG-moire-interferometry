# Example: binary amplitude grating

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalize

#Sim parameters
N = 2048 # number of grid points
L = 700e-6 # total size of the grid [m]
wvl = .6238e-6 # optical wavelength [m]
P = 40e-6
z = 1e-5 # propagation distance [m]

z_talbot = 2*P**2/wvl

# Sim computation
wave = OptWave(N,L,wvl)
wave.planeWave()

# Element
wave.rectAmplitudeGrating(P)
plt.plot(wave.x,np.abs(wave.U))
plt.show()

# Propagation
#wave.fraunhofer(z)
#wave.fresnel_DI(z)
#wave.fresnel_CV(z)
wave.fresnel_AS(z_talbot/4, simpson=False)
#wave.rayleigh_sommerfeld(z, fast=False)
#wave.angular_spectrum_repr(z_talbot/4, simpson=False)
#wave.bpm(z_talbot/4)

# Get results
xout = wave.x
Uout = wave.U

# Analytical solution
#Uan = wave.planeRectAmpGrAnalyticSolution(z,P,D)

# Normalized ntensity
Iout = normalize(np.abs(Uout))**2
#Ian = normalize(np.abs(Uan))**2

# Plot results
plt.plot(xout, Iout, "-")
#plt.plot(xout, Ian, "--")
plt.show()

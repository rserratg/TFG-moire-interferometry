# General example

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave

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
#wave.rectAperture(D)
#wave.doubleSlit(1e-3, D)
wave.rectPhaseGrating(0.5e-3, np.pi)

# Propagation
#wave.fraunhofer(z)
#wave.fresnel_DI(z)
#wave.fresnel_CV(z)
#wave.fresnel_AS(z, 4e4)
#wave.rayleigh_sommerfeld(z, fast=False)
#wave.angular_spectrum_repr(z)

# Get results
xout = wave.x
Uout = wave.U

# Analytical solution
#Uan = wave.planeSinAmpGrAnalyticSolution(z,1,5000,D)
#Uan = wave.planeDoubleSlitAnalyticSolution(z,1e-3,D)
#Uan = wave.planeRectAmpGrAnalyticSolution(z,D,7.5e-2)
#Uan = wave.planeRectAnalyticSolution(z, D)
Uan = wave.planeRectFresnelSolution(z,D)

# Plot results

# Notes on numpy: 
# Magnitude: np.abs(U)
# Phase: np.angle(U)

plt.plot(xout, np.angle(Uout), "-")
#plt.plot(xout, np.abs(Uan), "--")
plt.show()

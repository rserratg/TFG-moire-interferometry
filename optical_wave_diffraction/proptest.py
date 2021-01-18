# Code for propagation testing

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave

#Sim parameters
N = 1000# number of grid points
L = 4e-3 # total size of the grid [m]
wvl = 1e-6 # optical wavelength [m]
D = 0.5e-3
z = .1 # propagation distance

# Sim computation
wave = OptWave(N,L,wvl)
wave.planeWave()

# Element
#wave.rectAmplitudeGrating(0.2e-3)
wave.rectAperture(D)
#wave.doubleSlit(1e-3, D)
#wave.rectPhaseGrating(np.pi,0.2e-3)

# Propagation
wave.test_prop(z, method='AS')

# Get results
xout = wave.x
Uout = wave.U

# Plot results

# Notes on numpy: 
# Magnitude: np.abs(U)
# Phase: np.angle(U)

plt.plot(xout, np.abs(Uout), "-")
plt.show()

# Notes for AS: 
# Limiting wavelength provides good results (avoid fluctuations, but zero-padding needed for sides)
# Setting z small enough or L large enough, results are good (even w/o zero-padding)
# Haven't managed to apply zero-padding

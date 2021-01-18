# Example: 
# Plane wave (unit amplitude) incident on rectangular aperture
# Fresnel propagation via convolution theorem

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave

#Sim parameters
N = 4096 # number of grid points
L = 1e-2 # total size of the grid [m]
wvl = 1e-6 # optical wavelength [m]
D = 2e-3 # diamater of the aperture [m]
z = 1 # propagation distance

# Sim computation
wave = OptWave(N,L,wvl)
wave.planeWave()
wave.rectAperture(D)
wave.fresnel_AS(z) # Notice: must be bandlimited for proper results

# Get results
xout = wave.x
Uout = wave.U
Uan = wave.planeRectFresnelSolution(z,D)

# Plot results
plt.plot(xout, np.abs(Uout), "-")
plt.plot(xout, np.abs(Uan), "--")
plt.show()

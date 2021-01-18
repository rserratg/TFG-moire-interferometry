## Angular Spectrum Error Analysis
#
#
# I suspect that there is no issue with your method, and instead that 
# the error you are observing is a consequence of the DFT itself. Since DFT is
# only an approximation of the Fourier transform, we should see the error 
# between the analytical solution and ang_spec decrease as we increase L
# and decrease \delta X.

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave

#Sim parameters
N_list = 1024*np.arange(1,20) # number of grid points
L_list = 1e-2*np.arange(1,40) # total size of the grid [m]
wvl = 1e-6 # optical wavelength [m]
D = 2e-3 # diamater of the aperture [m]
z = 1 # propagation distance

# Sum the difference between the magnitudes of each solution
magnitude_errors = ()
for N in N_list:
    for L in L_list:
        # Sim computation
        wave = OptWave(N,L,wvl)
        wave.planeWave()
        wave.rectAperture(D)
        wave.fresnel_ang_spec(z)
        
        # Get results
        xout = wave.x
        Uout = wave.U
        Uan = wave.planeRectFresnelSolution(z,D)
        
        magnitude_errors += (np.sum(np.abs(Uout)-np.abs(Uan)), )

magnitude_errors = np.array(magnitude_errors).reshape([len(N_list), len(L_list)])

# Plot results
[x,y] = np.meshgrid(N_list, L_list)
plt.pcolormesh(x.transpose(), y.transpose(), np.log(magnitude_errors))
plt.xlabel("Number of points")
plt.ylabel("Space width [m]")
cbar = plt.colorbar()
cbar.set_label("Log(error) [arb. units]")
plt.title("Numerical Uncertainty with Simulation Paramters")
plt.show()


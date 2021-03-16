# Simulation of Lau carpet

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity, normalize

# Sim parameters
# With this parameters simulation takes around 1h15min
N = 10000
L = 20e-3
wvl = 589e-9
P = 200e-6
f = 0.1
angmax = 0.05 # maximum incidence angle
angnum = 101 # num of incidence angles

zmax = 75e-3
dz = 0.1e-3

# z_talbot = 0.135
# zmax is a bit more than z_talbot/2


# Output field from two-grating system in Lau effect
# z = distance between gratings = distance between 2nd grating and observation plane
# Total distance = 2z
def lau_output(z):
    print(z)
    
    I = np.zeros(N)
    for ang in np.linspace(-angmax, angmax, angnum):
        wave = OptWave(N,L,wvl)
        wave.planeWave(theta=ang)
        wave.rectAmplitudeGrating(P,f)
        wave.angular_spectrum(z)
        wave.rectAmplitudeGrating(P,f)
        wave.angular_spectrum(z)
        I += intensity(wave.U)
    return normalize(I)
        
# Simulation
z = 0
zticks = []
I = []

while z <= zmax:
    Iz = lau_output(z)
    I.append(Iz)
    zticks.append(z)
    z += dz

# Get x-space
waveAux = OptWave(N,L,wvl)
x = waveAux.x    

# Fix colormesh axes
zticks.append(z)
x = np.concatenate((x, [-x[0]]))

# Convert python lists to numpy arrays
zticks = np.asarray(zticks)
I = np.asanyarray(I)
       
# Plot intensity w.r.t. x and z
plt.pcolormesh(x*1e6, zticks*1e3, I)
plt.xlim(-500, 500)
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("z [mm]")
clb = plt.colorbar()
clb.set_label("Intensity [arbitrary units]")
plt.tight_layout()
plt.show()

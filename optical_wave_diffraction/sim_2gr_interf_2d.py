# Simulation - 2 grating interferometer
# Simuations to compare with experimental results

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave2D
from optwavepckg.utils import normalizedIntensity

# SIMULATION SETTINGS

print('2 grating moiré simulation (2D)')

print('Setting parameters...')

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L1 = 40e-3
L2 = 150e-3
D = 20e-3
L3 = 150e-3

# Grating settings
P = 180e-6
phi = np.pi/2

# Lens 
f = -25e-3

# Numerical parameters
N = 1024*10
M = 1000
Lx = 30e-3
Ly = 10e-3

# Run sim

print('Running simulation...')

wave = OptWave2D((N,M), (Lx,Ly), wvl)
wave.gaussianBeam(w0, z0=z0)

print('Propagation to lens...')

wave.angular_spectrum(L1)
wave.lens(f)

print('Propagation to first grating...')

wave.angular_spectrum(L2)
wave.rectPhaseGratingX(P, phi)

print('Propagation to second grating...')

wave.angular_spectrum(11.1e-3)
#wave.rectPhaseGratingX(P, phi)

print('Propagation to camera...')

#wave.angular_spectrum(L3)

# Get results
x = wave.x
y = wave.y
I = normalizedIntensity(wave.U)

# Plot
print('Plotting...')

# Resample less points
#stepx = int(N/1e4)
stepy = int(M/1e2)
#x = x[::stepx]
y = y[::stepy]
I = I[::stepy,:]

plt.plot(x, I[y==0].flatten())
plt.show()

exit()

plt.pcolormesh(x, y, I, shading='nearest')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('scaled')
clb = plt.colorbar()
clb.set_label('Intensity [arbitrary units]')
plt.show()



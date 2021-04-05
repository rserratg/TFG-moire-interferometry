# Two phase-grating moire interferometer
# Simulations to compare with experimental results

# 2D images at camera
# D constant
# Changing distance from G2 to camera
# Total distance measured from lens to camera


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

Lmin = 20e-2
Lmax = 36e-2
Lstep = 2e-2
Lvals = np.arange(Lmin, Lmax+Lstep, Lstep)

# Grating settings
P = 180e-6
phi = np.pi/2

# Lens 
f = -25e-3

# Numerical parameters
N = 1024*8
M = 1024
Lx = 30e-3
Ly = 5e-3

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

wave.angular_spectrum(D)
wave.rectPhaseGratingX(P, phi)

print('Storing intermediate results...')

# Warning: u0 = wave.U as long as wave.U is not modified
# For safety, it would be better to use a copy (or deepcopy)
# We can use it here because we apply np.fft.fft, which returns a new object
x = wave.x
y = wave.y
u0 = wave.U

print('Propagations to camera')

# We do it this way to avoid rounding errors in L3 <= L3max
for L in Lvals:
    print(L)

    wave.U = u0   
    wave.angular_spectrum(L-L2-D)

    I = normalizedIntensity(wave.U)
    
    # Reduce points in plot
    xmax = 5e-3
    condx = (x > -xmax) & (x < xmax) 
    xaux = x[condx]
    I = I[:,condx]

    plt.figure()
    plt.pcolormesh(xaux*1e3, y*1e3, I, shading='nearest')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('scaled')
    #plt.xlim(-4.86/2, 4.86/2) # Camera size
    #plt.ylim(-3.615/2, 3.615/2)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    #plt.show()
    
    plt.savefig(f"./plots/PGMI2_sims/2D/PGMI2_2d_{int(L*1e2)}_camsize.png", dpi=400)

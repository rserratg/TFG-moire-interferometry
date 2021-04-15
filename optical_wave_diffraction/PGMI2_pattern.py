# 2PGMI - Pattern at camera (1D or 2D)
# Monochromatic beam

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave, OptWave2D
from optwavepckg.utils import normalizedIntensity, intensity

# 1 = 1d
# 2 = 2d
numsim = 1

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L0 = 11e-2
L1 = 32e-2
D = 2.7e-2
L = 1

# Grating settings
P = 180e-6
phi = np.pi/2

# Lens 
f = -75e-3

def pattern():
    
    # Numerical parameters
    N = 1e6
    Lx = 60e-3

    # Run sim

    print('Running simulation...')

    wave = OptWave(N, Lx, wvl)
    wave.gaussianBeam(w0, z0=z0)

    print('Propagation to lens...')

    wave.angular_spectrum(L0)
    wave.lens(f)

    print('Propagation to first grating...')

    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi)

    print('Propagating to second grating...')

    wave.angular_spectrum(D)
    wave.rectPhaseGrating(P, phi)
    
    print('Propagating to camera...')
    
    wave.angular_spectrum(L-L1-D+f)

    print('Plotting...')

    x = wave.x
    I = intensity(wave.U)
    
    # Reduce points in plot
    xmin = -10e-3
    xmax = 10e-3
    condx = (x >= xmin) & (x <= xmax) 
    x = x[condx]
    I = I[condx]

    plt.figure()
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.show()
    
    #plt.savefig(f"./plots/Tests/PGMI2/PGMI2_{int(D*1e2)}.png")


def image():

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

    wave.angular_spectrum(L0)
    wave.lens(f)

    print('Propagation to first grating...')

    wave.angular_spectrum(L1)
    wave.rectPhaseGratingX(P, phi)

    print('Propagatiing to second grating...')

    wave.angular_spectrum(D)
    wave.rectPhaseGratingX(P, phi)
    
    print('Propagating to camera...')
    
    wave.angular_spectrum(L-L1-D+f)

    print('Plotting...')

    x = wave.x
    y = wave.y
    I = normalizedIntensity(wave.U)
    
    # Reduce points in plot
    xmin = -12.5e-3
    xmax = 5e-3
    condx = (x >= xmin) & (x <= xmax) 
    xaux = x[condx]
    I = I[:,condx]

    plt.figure()
    plt.pcolormesh(xaux*1e3, y*1e3, I, shading='nearest')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('scaled')
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    #plt.show()
    
    plt.savefig(f"./plots/Tests/PGMI2/PGMI2_2d_{int(D*1e2)}.png", dpi=400)
    

if numsim == 1:
    pattern()
elif numsim == 2:
    image()
else:
    print('Invalid sim number')
    exit()

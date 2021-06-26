# Script to generate plots for Journal - 2d images of talbot effect under cone beam

# List of plots
# 1. Pi, 2cm
# 2. Pi, 3cm
# 3. Pi, 4cm
# 4. Pi, 5.5cm
# 5. Pi/2, 2cm
# 6. Pi/2, 3cm
# 7. Pi/2, 4cm
# 8. Pi/2, 5.5cm

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave2D
from optwavepckg.utils import intensity

# Number of plot to calculate
# To save all, choose -1
numplot = 7

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L0 = 40e-3
L1 = 150e-3

# Grating settings
P = 180e-6

# Lens
f = -25e-3

# Numerical parameters
N = 1024 * 8
M = 1024
Lx = 30e-3
Ly = 5e-3

def generate_plot(phi, z):

    phase = 'Pi' if phi==np.pi else 'Pi2'
    d = z*1e2
    print('Params:', phase, d)
    
    wave = OptWave2D((N,M), (Lx,Ly), wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGratingX(P, phi)
    wave.angular_spectrum(z)
    
    x = wave.x
    y = wave.y
    I = intensity(wave.U)
    
    # Reduce points in plot
    xmax = 5e-3
    condx = (x > -xmax) & (x < xmax)
    x = x[condx]
    I = I[:,condx]
    
    plt.figure()
    plt.pcolormesh(x*1e3, y*1e3, I, shading='nearest')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('scaled')
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    
    #plt.show()
    
    # Watch out for aliasing issues in the plot! (must use high enough dpi)
    plt.savefig(f"./plots/Talbot_conebeam_2d/Sim_{phase}_{d:.2f}cm.png", dpi=800)
    
# Main script

# Parameters for each plot
# For each plot: (phi, z)
params = {
    1: (np.pi, 2e-2),
    2: (np.pi, 3e-2),
    3: (np.pi, 4e-2),
    4: (np.pi, 5.5e-2),
    5: (np.pi/2, 2e-2),
    6: (np.pi/2, 3e-2),
    7: (np.pi/2, 4e-2),
    8: (np.pi/2, 5.5e-2),
}

if numplot == -1:
    for p in params.values():
        generate_plot(*p)
elif numplot in params:
    generate_plot(*params[numplot])
else:
    print('Invalid numplot')
    exit()



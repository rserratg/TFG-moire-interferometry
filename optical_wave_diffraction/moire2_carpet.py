# Test: two phase-grating Moiré interferometer - Moiré carpet

'''
    Experiment from Miao_2016.
    
     - Distance from source to screen fixed at L=20cm
     - Gratings placed midway between them
     - Inter-grating spacing D in the range between 0.05mm and 5mm
'''

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import intensity, normalizedIntensity

# Sim parameters
N = 300001
L = 20e-3
wvl = 0.55e-6

# Source parameters
W = 0.44e-3 # source width
f = -5e-3 # lens focal for cone beam

# System parameters
P = 14.4e-6
Lt = 20e-2 # distance from source to screen
L1 = 10e-2
D = np.linspace(0.05e-3, 5e-3, 100)
phi = np.pi/2

# Calculate field right after first grating
wave = OptWave(N,L,wvl)
wave.gaussianBeam(W)
wave.lens(f)
wave.angular_spectrum_repr(L1)
wave.rectPhaseGrating(P, phi)

# Store field 
# This allows to run the first propagation only once
x = wave.x
u = wave.U
I = []
mask = np.abs(x) <= 10e-3 # central part of image

# Test for different values of D
# Last value of D is not calculated because pcolormesh will not plot it
for d in D[:-1]:
    print(d)
    wave.U = u
    wave.angular_spectrum_repr(d, simpson=False)
    wave.rectPhaseGrating(P, phi)
    wave.angular_spectrum_repr(Lt-L1-d)
    Id = normalizedIntensity(wave.U)
    
    # Take only central data
    I.append(Id[mask])


# Plot carpet
I = np.asanyarray(I)
xaux = x[mask]
plt.pcolormesh(xaux*1e3, D*1e3, I)
clb = plt.colorbar()
plt.xlabel('x [mm]')
plt.ylabel('D [mm]')
clb.set_label('Intensity [arbitrary units]')
plt.tight_layout()
plt.show()



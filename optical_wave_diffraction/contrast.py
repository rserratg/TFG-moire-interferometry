# Test visibility function 
# Plot contrast as function of z
# Results agree with Torcal-Milla 2009

# Test: L=10e-3, P=200e-6, wvl=0.6328e-6, N = 500001, phi/2 gr

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import intensity, visibility
   

N = 500001  
L = 10e-3
wvl = 0.6328e-6

P = 200e-6


zt = 2*P**2/wvl

z = np.linspace(0, 5/4*zt, 100)
V = []

for zz in z:
    print(zz)

    wave = OptWave(N,L,wvl, symmetric=False)
    wave.planeWave()
    #wave.rectAmplitudeGrating(P)
    wave.rectPhaseGrating(P, np.pi/2)
    wave.angular_spectrum_repr(zz, simpson=False)
    Vz = visibility(intensity(wave.U), wave.x, P)
    V.append(Vz)
    
plt.plot(z, V)
plt.show()

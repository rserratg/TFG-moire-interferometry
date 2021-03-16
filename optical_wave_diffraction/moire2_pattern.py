# Test: two phase-grating Moiré interferometer - intensity pattern
'''
 Note: 
    Fringe period = P*L/D
    Contrast period = 2*P**2/wvl = ZT
    Max contrast at (2*n+1)*ZT/2 (aprox)
    For P=14.4 um, wvl=550 nm -> ZT/2 = 0.37 mm
    For P=200 um, wvl = 1.55 um -> ZT/2 = 25.8 mm
    Notice that in this case for n=1 the fringes period is already 2*P
''' 

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import normalizedIntensity, binning

N = 1000001
L = 60e-3
wvl = 1.55e-6

W = 0.44e-3
f = -5e-3

P = 200e-6
Lt = 20e-2
L1 = 10e-2
D = 24e-3
L2 = Lt-L1-D
phi = np.pi/2

wave = OptWave(N,L,wvl)
wave.gaussianBeam(W)
wave.lens(f)
wave.angular_spectrum(L1)
wave.rectPhaseGrating(P, phi)
wave.angular_spectrum(D)
wave.rectPhaseGrating(P, phi)
wave.angular_spectrum(L2)
I = normalizedIntensity(wave.U)

plt.plot(wave.x, I)
plt.show()

newI, newX = binning(I,wave.x,10000)
plt.bar(newX, newI, width=newX[1]-newX[0],align='edge')
plt.show()

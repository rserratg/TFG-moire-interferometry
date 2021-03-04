# Test: two phase-grating Moiré interferometer - intensity pattern

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalizedIntensity

N = 1000001
L = 60e-3
wvl = 0.55e-6

W = 0.44e-3
f = -5e-3

P = 14.4e-6
Lt = 20e-2
L1 = 10e-2
D = 0.4e-3
L2 = Lt-L1-D
phi = np.pi/2

wave = OptWave(N,L,wvl)
wave.gaussianBeam(W)
wave.lens(f)
wave.angular_spectrum_repr(L1)
wave.rectPhaseGrating(P, phi)
wave.angular_spectrum_repr(D)
wave.rectPhaseGrating(P, phi)
wave.angular_spectrum_repr(L2)
I = normalizedIntensity(wave.U)

plt.plot(wave.x, I)
plt.show()

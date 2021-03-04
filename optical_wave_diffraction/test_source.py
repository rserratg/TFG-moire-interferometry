# Test: source diffraction angle / cone beam

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import intensity, normalizedIntensity

N = 50001
L = 40e-3
wvl = 1.55e-6
P = 180e-6

# Source / cone beam
W = 0.44e-3 # source slit width
f = -5e-3 # lens focal

wave = OptWave(N,L,wvl)
#wave.planeWave()
#wave.rectAperture(W)
wave.gaussianBeam(W)
wave.angular_spectrum_repr(30e-2)
wave.lens(f)

u = wave.U
x = wave.x


zticks = np.linspace(0, 20e-2, 100)
I = []

for z in zticks[:-1]:
    wave.U = u
    wave.angular_spectrum_repr(z)
    I.append(normalizedIntensity(wave.U))
    
I = np.asanyarray(I)
xaux = np.concatenate((x, [-x[0]]))

plt.pcolormesh(xaux, zticks, I)
clb = plt.colorbar()
plt.tight_layout()
plt.show()



wave.angular_spectrum_repr(20e-2)
I = normalizedIntensity(wave.U)
plt.plot(x, I)
plt.show()

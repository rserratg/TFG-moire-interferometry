import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import normalizedIntensity

wvl = 1e-6
L = 4e-3
N = 1001

z = .02

wave = OptWave(N,L,wvl)
wave.planeWave()
wave.rectPhaseGrating(0.1e-3, np.pi/2)
wave.angular_spectrum(z)


x = wave.x
I = normalizedIntensity(wave.U)

plt.plot(x*1e3, I)
plt.xlabel('x [mm]')
plt.ylabel('Intensity')
plt.show()

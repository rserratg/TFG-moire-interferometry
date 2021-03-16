import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave2D
from optwavepckg.utils import normalizedIntensity

N = 1001
L = 4e-3
wvl = 1e-6
D = 0.5e-3
z = .02

wave = OptWave2D(N,L,wvl)
wave.planeWave()
wave.rectPhaseGratingX(0.1e-3, np.pi/2)
wave.angular_spectrum(z)

x = wave.x
I = wave.U[wave.y==0].flatten()

I = normalizedIntensity(I)

plt.plot(x, I)
plt.show()

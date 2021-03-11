import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import intensity

N = 500
L = 20e-3
wvl = 1e-6

z = 1

wave = OptWave(N,L,wvl)
wave.planeWave()
wave.rectAmplitudeGrating(1e-3)
wave.rectAperture(10e-3)
wave.fraunhofer(z)

x = wave.x
u = wave.U
I = intensity(u)

uan = wave.planeRectAmpGrAnalyticSolution(z, 1e-3, 10e-3)
Ian = intensity(uan)

plt.plot(x, I)
plt.plot(x, Ian, '--')
plt.show()

# Talbot interferometer
# Phase object imaging
# Setup from Ibarra 1992: "Talbot interferometry: a new geometry"

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import intensity, visibility
   

N = 500001
L = 10e-2
wvl = 0.6328e-6

P = 3.175e-4
zt = 2*P**2/wvl # 31.86cm

f = 25e-2 # focal length of imaging lens
D = 0.05e-2 # width of spatial filter (slit)

wave = OptWave(N,L,wvl)
wave.planeWave()
wave.rectAmplitudeGrating(P,x0=0)
wave.angular_spectrum_repr(zt/4, simpson=True)
# object
wave.trapezoidPhaseObject(np.pi, 1e-2, 1.1e-2)
wave.angular_spectrum_repr(zt/4)
wave.rectAmplitudeGrating(P)
wave.angular_spectrum_repr(2*f-zt/4)
wave.lens(f)
wave.angular_spectrum_repr(f)
wave.rectAperture(D)
wave.angular_spectrum_repr(f)
    
x = wave.x
I = intensity(wave.U)

#print(visibility(I,x,P))

#plt.plot(x, np.angle(wave.U))
plt.plot(x, I)
plt.show()

#print(zt)

# Lau effect

# Not good results

# L = 2*b original field size
# 2*a observation screen size
# z propagastion distance
# amax = arctg((b-a)/z)
# step must be sufficiently small

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalize

N = 4000
L = 20e-3
wvl = 589e-9
P = 200e-6

z_talbot = 2*P**2/wvl
print(z_talbot)

I = np.zeros(N)
wave = OptWave(N,L,wvl)
x = wave.x

amax = 0.00
ang=amax

#for ang in np.linspace(-amax, amax, 201):
wave = OptWave(N,L,wvl)
wave.planeWave(theta=ang)
wave.rectAmplitudeGrating(P, 0.1)
wave.angular_spectrum_repr(z_talbot)
#wave.rectAmplitudeGrating(P, 0.1)
#wave.angular_spectrum_repr(z_talbot)
I += np.abs(wave.U)**2
    #break
    
I = normalize(I)



# Plot intensity at screen
plt.plot(x, I)
#plt.xlim(-500e-6, 500e-6)
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()

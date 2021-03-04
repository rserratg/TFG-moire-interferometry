# Simulation of Lau effect for an amplitude grating

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import intensity, normalize

# Sim parameters
N = 50000
L = 20e-3
wvl = 589e-9
P = 200e-6
f = 0.1 # duty cycle
angmax = 0.05 # max angle of incidence in radians

z_talbot = 2*P**2/wvl

print("Z Talbot: ", z_talbot)

I = np.zeros(N)
Igr = np.zeros(N)

for ang in np.linspace(-angmax,angmax, 101):
    
    print(ang)

    wave = OptWave(N,L,wvl)
    wave.planeWave(theta=ang)
    wave.rectAmplitudeGrating(P, f)
    Igr += intensity(wave.U)
    
    wave.angular_spectrum_repr(z_talbot/4)
    wave.rectAmplitudeGrating(P, f)
    wave.angular_spectrum_repr(z_talbot/4)
    I += intensity(wave.U)
    #break

# Get x-space
wave = OptWave(N,L,wvl)
x = wave.x

plt.plot(x, normalize(Igr))
plt.plot(x, normalize(I))
plt.xlim(-500e-6, 500e-6)
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()

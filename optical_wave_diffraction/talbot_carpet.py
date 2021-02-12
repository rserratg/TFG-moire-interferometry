# Talbot carpet for an amplitude or phase grating

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalize

'''

# Sim parameters - set 1
# x-space plot: -150e-6, 150e-6
N = 20000 # number of points
L =  2000e-6 # grid size
wvl = .6238e-6 # wavelength
P = 40e-6 # grating period
ff = 0.5 # grating duty cycle
zmax = 10e-3 # total propagation distance
dz = 10e-6 # propagation step distance

'''

# Sim parameters - set 2
# x-space plot: -500e-6 ,500e-6
N = 30000
L = 20e-3
wvl = 532e-9
P = 200e-6
ff = 0.5
zmax = 200e-3
dz = 50e-6

#'''

# Talbot distance
z_talbot = 2*P**2/wvl # = 5.13e-3 / 150mm

'''

z = 0
zticks = []
I = []
x = np.empty(N)

while z <= zmax:
    wave = OptWave(N,L,wvl)
    wave.planeWave()
    #wave.rectAmplitudeGrating(P, ff)
    wave.rectPhaseGrating(P, np.pi)
    #wave.sinAmplitudeGrating(1, 1/P, L)
    
    if z == 0:
        x = wave.x
    
    wave.angular_spectrum_repr(z, simpson=False)
    #wave.fresnel_AS(z, simpson=False)
    Iz = normalize(np.abs(wave.U))**2
        
    zticks.append(z)
    I.append(Iz)
    z += dz
      
# Colormesh: ideally x and zticks dimensions should be 1 greater than those of I
# Otherwise the last row/column of I will be ignored
# Notice that x is not simetric
zticks.append(z)
x = np.concatenate((x, [-x[0]]))

# Convert python lists to numpy arrays
zticks = np.asarray(zticks)
I = np.asanyarray(I)
     
# Plot intensity w.r.t. x and z
plt.pcolormesh(x, zticks, I)
#plt.xlim(-150e-6, 150e-6)
plt.xlim(-500e-6,500e-6)
plt.xlabel('x [m]')
plt.ylabel('z [m]')
clb = plt.colorbar()
clb.set_label('Intensity [arbitrary units]')
plt.title("Talbot amplitude grating: ff={}".format(ff))
plt.tight_layout()
plt.show()

'''

# Plot intensity pattern at talbot distance

wave2 = OptWave(N,L,wvl)
wave2.planeWave()
wave2.rectAmplitudeGrating(P, ff)
#wave2.rectPhaseGrating(P, np.pi, ff=0.25)
#wave2.sinAmplitudeGrating(1, 1/P, L)

#wave2.fresnel_AS(z_talbot*3/4, simpson=False)
wave2.angular_spectrum_repr(z_talbot/4, simpson=False)

# ---
wave3 = OptWave(N,L,wvl)
wave3.planeWave()
#wave3.rectPhaseGrating(P, np.pi, ff=0.25)
wave3.rectAmplitudeGrating(P,ff)
wave3.fresnel_AS(z_talbot/4, simpson=False)
# ---


#plt.plot(wave2.x, np.angle(wave2.U), "-")
plt.plot(wave2.x, normalize(np.abs(wave2.U))**2, "-")
plt.plot(wave2.x, normalize(np.abs(wave3.U))**2, "--")

#plt.xlim(-150e-6, 150e-6)
plt.xlim(-500e-6, 500e-6)
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
#plt.title("Talbot amplitude grating: z=zt, ff={}".format(ff))
plt.show()

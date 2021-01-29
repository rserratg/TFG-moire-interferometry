# Example: Talbot carpet

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalize

# Sim parameters
N = 2000 # number of points
L = 2000e-6 # grid size
wvl = .6238e-6 # wavelength
P = 40e-6 # grating period
zmax = 10e-3 # total propagation distance
dz = 1e-5 # propagation step distance

# Talbot distance
z_talbot = 2*P**2/wvl # = 5.13e-3

z = 0
zticks = []
I = []
x = np.empty(N)
I_talbot = np.zeros(N)

while z <= zmax:
    wave = OptWave(N,L,wvl)
    wave.planeWave()
    #wave.rectAmplitudeGrating(P, ff=0.25)
    wave.rectPhaseGrating(np.pi/2, P)
    
    if z == 0:
        x = wave.x
    
    wave.angular_spectrum_repr(z, simpson=False)
    Iz = normalize(np.abs(wave.U))**2
    
    if abs(z - z_talbot) <= dz/2:
        I_talbot = Iz
        
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
plt.xlim(-150e-6,150e-6)
plt.xlabel('x [m]')
plt.ylabel('z [m]')
clb = plt.colorbar()
clb.set_label('Intensity [arbitrary units]')
plt.title("Talbot amplitude grating: ff=0.25")
plt.tight_layout()
plt.show()


# Plot intensity pattern at talbot distance

wave2 = OptWave(N,L,wvl)
wave2.planeWave()
#wave2.rectAmplitudeGrating(P, ff=0.25)
wave2.rectPhaseGrating(np.pi/2, P)
wave2.angular_spectrum_repr(z_talbot/2, simpson=False)

#plt.plot(wave2.x, np.angle(wave2.U), "-")
plt.plot(wave2.x, normalize(np.abs(wave2.U))**2, "-")
#plt.plot(wave2.x, I_talbot, "--")

plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
plt.title("Talbot amplitude grating: z=zt, ff=0.25")
plt.show()

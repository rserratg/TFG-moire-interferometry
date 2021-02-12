# Talbot effect for Gaussian beam

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalizedIntensity

# Sim parameters
# x-space plot: -500e-6 ,500e-6
N = 10000
L = 2000e-6
wvl = 623.8e-9
P = 15e-6
ff = 0.5

w0 = 150e-6
zr = w0**2*np.pi/wvl

# Talbot distance for gaussian beam
# Assuming z0=0 -> R=inf, w=w0
nu = 10 # order of distance
aux = nu*P**2/wvl
z = (1 - np.sqrt(1-4*aux**2/zr**2))*zr**2/aux/2
print(z)


# Talbot distance (for comparison)
z_talbot = nu*P**2/wvl
print(z_talbot)

print("zr:", zr)

wave = OptWave(N,L,wvl)
wave.gaussianBeam(w0=w0)
wave.rectAmplitudeGrating(P)
#wave.rectPhaseGrating(P, phi=np.pi)

# Plot intensity right after grating
#plt.plot(wave.x, normalizedIntensity(wave.U))
#plt.xlim(-300e-6, 300e-6)
#plt.show()

# Propagate
# To see orders of diffraction: far field - around 10mm
# To see self-images: near field - z (nu = 1 or 2 for amp gr, 3/2 for pi/2 phase gr)
wave.angular_spectrum_repr(z)

# Plot intensity at observation plane
plt.plot(wave.x, normalizedIntensity(wave.U))
#plt.xlim(-300e-6, 300e-6)
plt.show()

'''
List of plots:
    - Talbot_gauss_zt: amplitude gr, nu=1 (z=0.0007214142983190331, z_t/2=0.0007213850593138826)
    - Talbot_gauss_zt_ampli: amplitude gr, nu=2, blue=AS, orange=Fresnel-AS
    - Talbot_gauss_2zt: amplitude gr, nu=2
    - Talbot_gauss_far: amplitude gr, z=10mm
    - Talbot_gauss_pi2_3zt2: pi/2 phase gr, nu=3/2
    - Talbot_gauss_pi2_far: pi/2 phase gr, z=10mm
    - Talbot_gauss_pi_far: pi phase gr, z=5mm
'''

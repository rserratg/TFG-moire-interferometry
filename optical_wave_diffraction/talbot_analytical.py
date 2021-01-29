# Example: Talbot carpet for amplitude grating with analytical solution

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalize

'''
    Fourier coefficients of binary amplitude grating (An).
    
    Parameters:
        - n
        - f: opening fraction / duty cycle / fill factor
'''
def fourier_coef_amp_gr(n, f):
    if n == 0:
        return f
    else:
        return np.sin(n*np.pi*f)/(n*np.pi)
        
'''
    Wave after binary amplitude grating
    
    Parameters:
        - x (numpy.array): x-space
        - wvl: wavelength
        - z: propagation distance
        - p: grating period
        - f: opening fraction of grating
        - nmax: maximum order for fourier series of grating
'''
def talbot_wave(x, wvl, z, p, f, nmax):
    u = np.zeros_like(x, dtype=complex)
    
    kp = 2 * np.pi / p
    Lt = p**2/wvl
    
    for n in range(-nmax,nmax+1):
        An = fourier_coef_amp_gr(n, f)
        phase1 = np.exp(1j*n*kp*x)
        phase2 = np.exp(-1j*n**2*np.pi*z/Lt)
        u += An * phase1 * phase2
        
    return u
    
'''
    Normalized intensity of field
    
    Parameters:
        - u: complex field
'''
def intensity(u):
    return normalize(np.abs(u))**2
    
# -------------------------------

# Sim parameters
N = 2000 # number of points
L = 2000e-6 # grid size
wvl = .6238e-6 # wavelength
P = 40e-6 # grating period
f = 0.1 # grating duty cycle
zmax = 10e-3 # total propagation distance
dz = 1e-5 # propagation distance step
nmax = 25 # maximum order for grating fourier series

z = 0
zticks = []
I = []
x = np.arange(-N//2, N//2)*(L/N)

while z <= zmax:
    uz = talbot_wave(x, wvl, z, P, f, nmax)
    Iz = intensity(uz)
    
    zticks.append(z)
    I.append(Iz)
    z += dz
    
# Adapt axis for colormesh
zticks.append(z)
xplt = np.concatenate((x, [-x[0]]))

# Convert python lists to numpy arrays
zticks = np.asarray(zticks)
I = np.asanyarray(I)

# Plot intensity w.r.t. x and z
plt.pcolormesh(xplt, zticks, I)
plt.xlim(-150e-6,150e-6)
plt.xlabel('x [m]')
plt.ylabel('z [m]')
clb = plt.colorbar()
clb.set_label('Intensity [arbitrary units]')
plt.title("Talbot amplitude grating: ff={}".format(f))
plt.tight_layout()
plt.show()
    
# Plot intensity pattern at talbot distance
zt = 2*P**2/wvl
I2 = intensity(talbot_wave(x, wvl, zt/8, P, f, nmax))
plt.plot(x, I2, "-")
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
plt.title("Talbot amplitude grating: z=zt, ff={}".format(f))
plt.show()

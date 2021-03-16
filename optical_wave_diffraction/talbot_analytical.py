# Talbot carpet for a grating using analytical solution

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import normalize

'''
    Fourier coefficients of binary amplitude grating (An).
    
    Parameters:
        - n (int): coefficient order
        - f (double): opening fraction / duty cycle / fill factor
'''
def fourier_coef_amp_gr(n, f):
    if n == 0:
        return f
    else:
        return np.sin(n*np.pi*f)/(n*np.pi)
        
'''
    Fourier coefficients of sine grating 
    T(x) = 1/2 + 1/2 * cos(2*pi*x/P)
    
    Parameters:
        - n (int): coefficient order
'''
def fourier_coef_sin_gr(n):
    if n == 0:
        return 1/2
    elif abs(n) == 1:
        return 1/4
    else:
        return 0
        
'''
    Fourier coefficients of binary phase grating
    
    T(x) = exp(-j*phi*G), where G = binary amplitude grating
    
    Parameters:
        - n (int): coefficient order
        - f (double): fill factor
        - phi (double): phase
'''
def fourier_coef_phase_gr(n, f, phi):
    if n == 0:
        return 1 - f + f * np.exp(-1j*phi)
    else:
        a = np.exp(-1j*phi) - 1
        b = n*np.pi
        return a*np.sin(b*f)/b
        
'''
    Wave after binary amplitude grating
    
    Parameters:
        - x (numpy.array): x-space
        - wvl: wavelength
        - z: propagation distance
        - p: grating period
        - nmax: maximum order for fourier series of grating
        - kind: grating type - 'amplitude', 'phase', 'sine'
        - grparams: (optional parameters)
            - f (double): fill factor for amp and phase gr
            - phi (double): phase for phase gr
'''
def talbot_wave(x, wvl, z, p, nmax, kind, **grparams):
    u = np.zeros_like(x, dtype=complex)
    
    kp = 2 * np.pi / p
    Lt = p**2/wvl
    
    for n in range(-nmax,nmax+1):
        
        if kind == 'amplitude': 
            An = fourier_coef_amp_gr(n, grparams["f"])
        elif kind == 'phase':
            An = fourier_coef_phase_gr(n, grparams["f"], grparams["phi"])
        elif kind == 'sine':
            if n < -1:
                continue
            elif n > 1:
                break
            An = fourier_coef_sin_gr(n)
        
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

kind = 'phase'
f = 0.25 # grating duty cycle
phi = np.pi

zmax = 10e-3 # total propagation distance
dz = 10e-6 # propagation distance step
nmax = 25 # maximum order for grating fourier series

z = 0
zticks = []
I = []
x = np.arange(-N//2, N//2)*(L/N)

#'''

while z <= zmax:
    uz = talbot_wave(x, wvl, z, P, nmax, kind=kind, f=f, phi=phi)
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
#plt.title("Talbot amplitude grating: ff={}".format(f))
plt.tight_layout()
plt.show()

#'''
    
# Plot intensity pattern at talbot distance
zt = 2*P**2/wvl
I2 = intensity(talbot_wave(x, wvl, zt, P, nmax, kind=kind, f=f, phi=phi))
plt.plot(x, I2, "-")
plt.xlim(-150e-6, 150e-6)
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
#plt.title("Talbot amplitude grating: z=zt, ff={}".format(f))
plt.show()

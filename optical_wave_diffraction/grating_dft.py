# Diffraction from an arbitrary periodic grating 
# Using Talbot theoretical result (see Case_2009)
# Grating fourier series calculated by DFT of a single period

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
    Wave after binary amplitude grating
    
    Parameters:
        - x (numpy.array): x-space
        - wvl: wavelength
        - z: propagation distance
        - p: grating period
        - nmax: maximum order for fourier series of grating
        - f (double): fill factor for amp and phase gr
'''
def talbot_wave(x, wvl, z, p, nmax, f):
    u = np.zeros_like(x, dtype=complex)
    
    kp = 2 * np.pi / p
    Lt = p**2/wvl
    
    nlist = [n for n in range(-nmax, nmax+1)]
    
    # Binary amplitude grating with analytical fourier coefs
    A = [fourier_coef_amp_gr(n,f) for n in nlist]
    
    # Binary amplitude gratings with transmission function of 1 period
    # Fourier series coeffs = 1/N * DFT
    B = np.where(np.abs(nlist)<=nmax*f, 1, 0)
    B = np.fft.ifftshift(B)
    B = np.fft.fft(B)
    B = np.fft.fftshift(B)
    B /= len(nlist)
    
    # Change A or B to use either method
    for i, n in enumerate(nlist):
        phase1 = np.exp(1j*n*kp*x)
        phase2 = np.exp(-1j*n**2*np.pi*z/Lt)
        u += B[i] * phase1 * phase2
       
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

f = 0.5 # grating duty cycle

zmax = 10e-3 # total propagation distance
dz = 10e-6 # propagation distance step
nmax = 100 # maximum order for grating fourier series

z = 0
zticks = []
I = []
x = np.arange(-N//2, N//2)*(L/N)

# Plot intensity pattern at talbot distance
zt = 2*P**2/wvl
I2 = intensity(talbot_wave(x, wvl, zt, P, nmax, f))
plt.plot(x, I2, "-")
plt.xlim(-150e-6, 150e-6)
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
#plt.title("Talbot amplitude grating: z=zt, ff={}".format(f))
plt.show()

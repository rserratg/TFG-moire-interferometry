# Lau effect for a grating using analytical solution

# Assuming plane wave
# More accurate results with lower angle sampling interval
# High computational cost

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalize, intensity

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
    Fourier coefficient for a given grating
    
    Parameters:
        - n (int): coefficient order
        - kind (string): gratind type - 'amplitude', 'phase', 'sine'
        - params (dictionary): grating parameters
            - f (double): duty cycle (amp/phase gr)
            - phi (double): phase (phase gr)
'''
def get_fourier_coef(n, kind, params):
    if kind == 'amplitude': 
        An = fourier_coef_amp_gr(n, params["f"])
    elif kind == 'phase':
        An = fourier_coef_phase_gr(n, params["f"], params["phi"])
    elif kind == 'sine':
        An = fourier_coef_sin_gr(n)
    return An
    
'''
    Wave after 1 grating
    Included for testing purposes
    
    Parameters:
        - x (numpy.array): x-space
        - wvl: wavelength
        - theta: wave angle of incidence (w.r.t normal of grating)
        - z: propagation distance between grating and observation plane
        - p: grating period
        - nmax: maximum order for fourier series of grating
        - kind: grating type - 'amplitude', 'phase', 'sine'
        - grparams: (optional parameters)
            - f (double): fill factor for amp and phase gr
            - phi (double): phase for phase gr
'''
def lau_wave_1gr(x, wvl, theta, z, p, nmax, kind, **grparams):
    u = np.zeros_like(x, dtype=complex)
    
    k = 2 * np.pi / wvl
    kp = 2 * np.pi / p
    kth = k * np.sin(theta)
    
    for n in range(-nmax,nmax+1):
        An = get_fourier_coef(n, kind, grparams)
        phase = np.exp(1j*(kth+n*kp)*x + 1j*(k-(kth+n*kp)**2/(2*k))*z)
        u += An * phase
        
    return u
    
'''
    Wave after 2 gratings.
    Assuming both gratings are equal
    
    Parameters:
        - x (numpy.array): x-space
        - wvl: wavelength
        - theta: wave angle of incidence (w.r.t normal of grating)
        - L1: propagation distance between first and second grating
        - L2: propgation distance between second grating and observation plane
        - p: grating period
        - nmax: maximum order for fourier series of grating
        - kind: grating type - 'amplitude', 'phase', 'sine'
        - grparams: (optional parameters)
            - f (double): fill factor for amp and phase gr
            - phi (double): phase for phase gr
'''
def lau_wave_2gr(x, wvl, theta, L1, L2, p, nmax, kind, **grparams):
    u = np.zeros_like(x, dtype=complex)
    
    k = 2 * np.pi / wvl
    kp = 2 * np.pi / p
    kth = k * np.sin(theta)
    
    for n in range(-nmax,nmax+1):
        An = get_fourier_coef(n, kind, grparams)
        if An==0:
            continue
            
        phase1 = np.exp(-1j*(kth+n*kp)**2*L1/(2*k))
        
        uaux = np.zeros_like(u, dtype=complex)        
        for m in range(-nmax, nmax+1):
            Am = get_fourier_coef(m, kind, grparams)   
            if Am==0:
                continue
            phase2 = np.exp(1j*(kth+(n+m)*kp)*x)
            phase3 = np.exp(-1j*(kth+(n+m)*kp)**2*L2/(2*k))
            uaux += Am * phase2 * phase3            
                
        u += An * phase1 * uaux
        
    return u                  
    
# -------------------------------

# Sim parameters
N = 10000 # number of points
L = 20e-3 # grid size
wvl = 589e-9 # wavelength
P = 200e-6 # grating period

z_talbot = 2*P**2/wvl

kind = 'amplitude'
f = 0.1 # grating duty cycle
phi = np.pi

L1 = z_talbot/2
L2 = L1
nmax = 20 # maximum order for grating fourier series

angles = np.linspace(-0.05, 0.05, 51)

x = np.arange(-N//2, N//2)*(L/N)
  
# Plot intensity pattern 

I = np.zeros_like(x)

for ang in angles:
    print("angle: {}".format(ang))
    u = lau_wave_2gr(x, wvl, ang, L1, L2, P, nmax, kind=kind, f=f, phi=phi)
    I += intensity(u)
    
I = normalize(I)

# Grating pattern for reference
wave = OptWave(N,L,wvl)
wave.planeWave()
wave.rectAmplitudeGrating(P, f)
plt.plot(wave.x, intensity(normalize(wave.U)))

plt.plot(x, I)
plt.xlim(-500e-6, 500e-6)
plt.xlabel("x [m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()

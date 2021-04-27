# 3PGMI - analytical intensity pattern
# (assuming monochromatic spherical wave)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

wvl = 1.55e-6
k = 2*np.pi/wvl

P = 180e-6
f1 = f2 = f3 = 1/P

L1 = 75e-3 + 32e-2
D1 = 10e-2
Lt = 1

dvals = np.linspace(-5e-2, 5e-2, 101)

y = np.linspace(-30e-3, 30e-3, 5001)

mmax = qmax = 2
nmax = 20

def contrast_FT(d, u, f0, fmax = None):

    # Calculate fourier transform and frequencies
    ft = np.abs(np.fft.rfft(u))
    freq = np.fft.rfftfreq(len(u), d)
    
    # Discard components above max freq
    if fmax is not None:
        cond = freq < fmax
        ft = ft[cond]
        freq = freq[cond]
    
    # Find indices of peaks
    # Maxima at f=0 and f=fmax are not considered as peaks
    pks, _ = find_peaks(ft)
    
    # Find peak closer to f0
    idx = (np.abs(freq[pks]-f0)).argmin()   # index of closer peak in pks
    idx = pks[idx]                          # index of closer peak in original arrays (ft,freq)
    
    # Results
    C = 2*ft[idx]/ft[0]
    fd = freq[idx]
    
    # Plot ft, position of peaks (x) and moire peak (red dot) (for debugging)
    if False:
        plt.plot(freq, ft, '-o')
        plt.plot(freq[pks], ft[pks], 'x')
        plt.plot(freq[idx], ft[idx], 'ro')
        plt.show()
    
    return C, fd

# Fourier coeffs for 50/50 phase grating
def coeffs(n, phi):
    if n == 0:
        return 1/2 + 1/2*np.exp(-1j*phi)
    else:
        return (np.exp(-1j*phi)-1)/(n*np.pi)*np.sin(n*np.pi/2)

# Pattern for a specific D3
def pattern(D3):

    L3 = Lt - D3 - D1 - L1
    Laux = L1 + D1 + D3

    Vy = np.zeros_like(y, dtype=np.complex128)

    for m in range(-mmax, mmax+1):
        
        Am = coeffs(m, np.pi/2)

        for n in range(-nmax, nmax+1):
        
            Bn = coeffs(n, np.pi)
            
            for q in range(-qmax, qmax+1):
            
                Cq = coeffs(q, np.pi/2)
                
                fd = m*f1*L1 + n*f2*(L1+D1) + q*f3*(L1+D1+D3)
                fd /= Lt
                
                # Source term not considered
                # fs = m*f1*(D1+D3+L3) + n*f2*(D3+L3) + q*f3*L3
                # fs /= Lt
                
                phi0 = (2*np.pi*m*f1)**2*L1/Laux + (2*np.pi*n*f2)**2*D3/Laux
                phi0 -= (2*np.pi*m*f1*L1/Laux - 2*np.pi*n*f2*D3/Laux)**2
                phi0 *= Laux/(2*k)
                
                phi1 = (m*f1*L1/Laux + n*f2*(L1+D1)/Laux + q*f3)**2
                phi1 *= 4*np.pi**2*L3*Laux/(2*k*Lt)
                
                Vy += Am*Bn*Cq * np.exp(1j*2*np.pi*fd*y) * np.exp(-1j*phi0 - 1j*phi1)
                
    return np.abs(Vy)**2
    
# Calculation

C = []
F = []

for D in dvals:
    print(f'D: {D}')
    D3 = D + D1
    Id = pattern(D3)
    fmoire = (1/P)*abs(D)/Lt
    c, fd = contrast_FT(y[1]-y[0], Id, fmoire)
    C.append(c)
    F.append(fd)
    
C = np.asarray(C)
F = np.asarray(F)
    
# Plot

fig, ax1 = plt.subplots()
    
color1 = 'tab:blue'
ax1.set_xlabel('D3 - D1 [mm]')
ax1.set_ylabel('Contrast', color=color1)
#ax1.set_ylim(0, 1)
ax1.plot(dvals*1e3, C, '-o', color=color1)

ax2 = ax1.twinx()

color2 = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color2)
ax2.plot(dvals*1e3, F*1e-3, 'x', color=color2)

# Theoretical frequency
Fd = (1/P)*np.abs(dvals)/Lt
ax2.plot(dvals*1e3, Fd*1e-3, '-', color=color2)

fig.tight_layout()
plt.show()

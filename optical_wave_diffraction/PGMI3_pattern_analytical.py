# 3PGMI - analytical intensity pattern
# (assuming monochromatic spherical wave)

import numpy as np
import matplotlib.pyplot as plt

wvl = 1.55e-6
k = 2*np.pi/wvl

P = 180e-6
f1 = f2 = f3 = 1/P

L1 = 75e-3 + 32e-2
D1 = 10e-2
D3 = 13e-2

Lt = 1
L3 = Lt - D3 - D1 - L1

Laux = L1 + D1 + D3

y = np.linspace(-10e-3, 10e-3, 501)

mmax = qmax = 20
nmax = 20

# Fourier coeffs for 50/50 phase grating
def coeffs(n, phi):
    if n == 0:
        return 1/2 + 1/2*np.exp(-1j*phi)
    else:
        return (np.exp(-1j*phi)-1)/(n*np.pi)*np.sin(n*np.pi/2)

# Calculation

Vy = np.zeros_like(y, dtype=np.complex128)

for m in range(-mmax, mmax+1):
    
    Am = coeffs(m, np.pi/2)

    for n in range(-nmax, nmax+1):
    
        Bn = coeffs(n, np.pi)
        
        for q in range(-qmax, qmax+1):
        
            Cq = coeffs(q, np.pi/2)
            
            print(m, n, q)
            
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
    
# Plot

I = np.abs(Vy)**2

ft = np.abs(np.fft.rfft(I))
freq = np.fft.rfftfreq(len(y), y[1]-y[0])
plt.plot(freq, ft, '-o')
plt.show()

plt.plot(y*1e3, I)
plt.xlabel('y [mm]')
plt.ylabel('Intensity')
plt.show()

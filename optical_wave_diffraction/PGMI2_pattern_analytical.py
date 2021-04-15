# 2PGMI - analytical intensity pattern
# (assuming monochromatic spherical wave)

import numpy as np
import matplotlib.pyplot as plt

wvl = 1.55e-6
k = 2*np.pi/wvl

P = 180e-6
f1 = f2 = 1/P

L1 = 75e-3 + 32e-2
D = 2.7e-2
L = 1
L2 = L - D - L1

y = np.linspace(-10e-3, 10e-3, 501)

mmax = nmax = 50


def coeffs(n):
    if n == 0:
        return 1/2 + 1/2*np.exp(-1j*np.pi/2)
    else:
        return (np.exp(-1j*np.pi/2)-1)/(n*np.pi)*np.sin(n*np.pi/2)



Vy = np.zeros_like(y, dtype=np.complex128)

for m in range(-mmax, mmax):
    
    Am = coeffs(m)

    for n in range(-nmax, nmax):
    
        print(m, n)
    
        Bn = coeffs(n)
        
        phi0 = (2*np.pi*m*f1*L1/L + 2*np.pi*n*f2*(L1+D)/L)*y
        phi1 = - L/(2*k)*( (2*np.pi*m*f1)**2*L1/L + (2*np.pi*n*f2)**2*L2/L - (2*np.pi*m*f1*L1/L - 2*np.pi*n*f2*L2/L)**2 )
        
        Vy += Am*Bn*np.exp(1j*phi0 + 1j*phi1)
        
I = np.abs(Vy)**2

plt.plot(y, I)
plt.xlabel('y [mm]')
plt.ylabel('Intensity')
plt.show()

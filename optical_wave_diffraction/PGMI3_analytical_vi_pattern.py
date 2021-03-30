# Test: analytical expression of virtual image in 3PGMI

import numpy as np
import matplotlib.pyplot as plt

# Fourier series coefs of binary pi/2 phase grating (50% duty cycle)
def pi2GratingCoef(m):
    if m == 0:
        return 1/2 + 1/2 * np.exp(-1j*np.pi/2)
    else:
        a = np.exp(-1j*np.pi/2) - 1
        b = m*np.pi
        return a*np.sin(b/2)/b


L = 5e-3
N = 2e5
y = np.linspace(-L/2, L/2, N)

wvl = 1.55e-6
k = 2*np.pi/wvl

P = 180e-6
f1 = f2 = 1/P

L1 = 75e-3 + 32e-2
D = 21e-2
L2 = D*f1/(2*f2-f1)
L = L1 + D + L2

# R = L / cos(theta)
R = np.sqrt(L**2 + y**2)

# Series orders
mmax = 50

# Pi phase grating coefs with 50% duty cycle (B_1 = B_-1)
B1 = -2/np.pi

Ve = 0

# Plot amplitude of each m term in the sum
#plt.figure()

for m in range(-mmax, mmax+1):
    
    Am = pi2GratingCoef(m)
    Am1 = pi2GratingCoef(m+1)
    
    phi0m = 2*np.pi*m*f1*(L1/L)*y 
    phi0n = 2*np.pi*f2*((L1+D)/L)*y
    phi0 = phi0m + phi0n
    
    phi1m = (2*np.pi*m*f1)**2*(L1/L)
    phi1n = (2*np.pi*f2)**2*(L2/L)
    phi1mn = (2*np.pi*m*f1*L1/L - 2*np.pi*f2*L2/L)**2
    phi1 = - L/(2*k) * (phi1m + phi1n - phi1mn)
    
    Vem = Am*B1 + Am1*B1*np.exp(1j*2*np.pi*(f1-2*f2)*y)
    Vem *= np.exp(1j * (phi0 + phi1))
    
    #plt.plot(y*1e3, np.real(Vem))
    
    Ve += Vem

#plt.xlabel('y [mm]')
#plt.ylabel('Amplitude [arb. units]')
#plt.show()

#Ve *= np.exp(1j*k*R)
I = np.abs(Ve)**2

plt.plot(y*1e3, I)
plt.xlabel('y [mm]')
plt.ylabel('Intensity')
plt.xlim(-2.5, 2.5)
plt.show()

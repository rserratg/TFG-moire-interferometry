# 2PGMI - theoretical frequency and contrast of moire fringes
# Using expression from Miao 2016

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS

# Setup params

wvl = 1.55e-6

P = 180e-6
f1 = f2 = 1/P

# L1 = distance from point source to first grating
L1 = 98e-2 + 75e-3
L = 208e-2

Dvals = np.linspace(1e-2,11e-2, 201)

# AUXILIARY FUNCTIONS

# Fourier series coeffs for pi/2 grating
def pi2coeffs(n):
    if n == 0:
        return 1/2 + 1/2*np.exp(-1j*np.pi/2)
    else:
        return (np.exp(-1j*np.pi/2)-1)/(n*np.pi)*np.sin(n*np.pi/2)
    
# SIMULATION

freq = []
cont = []
for D in Dvals:

    L2 = L - L1 - D
    
    Pd = L/((f2-f1)*L1+f2*D)
    freq.append(1/Pd)
    
    delta1 = wvl/L*f1*L1*((f1-f2)*L2+f1*D)
    delta2 = wvl/L*f2*L2*((f2-f1)*L1+f2*D)
    
    X1 = 0
    X2 = 0
    # Only terms m=-1 and m=0 are different than 0
    for m in range(-1, 1):
    
        Am = pi2coeffs(m)
        Am1 = pi2coeffs(m+1)
        
        # Note: X2 terms n, n-1 changed to n+1,n for faster calculation
        X1 += Am*np.conj(Am1)*np.exp(1j*2*np.pi*(m+1/2)*delta1)
        X2 += np.conj(Am)*Am1*np.exp(-1j*2*np.pi*(m+1/2)*delta2)  
    
    # Average intensity transmission through gratings = 1
    cont.append(np.abs(X1)*np.abs(X2))
    
freq = np.asarray(freq)
cont = np.asarray(cont)

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color)
ax1.plot(Dvals*1e3, 2*cont, color=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color)
ax2.plot(Dvals*1e3, freq*1e-3, color=color)

fig.tight_layout()
plt.show()

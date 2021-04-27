# 3PGMI - theoretical frequency and contrast of moire fringes
# Using expression from Miao 2016
# Using approximation: only orders +1 and -1 from second grating are considered

import numpy as np
import matplotlib.pyplot as plt

# Setup params

wvl = 1.55e-6

P = 180e-6
f1 = f2 = f3 = 1/P

# L1 = distance from point source to first grating
L1 = 75e-3 + 32e-2
D1 = 30e-2
L = 2

'''
    Fourier series coeffs for binary phase grating
    (exponential series)
    
    Parameters:
        - n (int): order of coeff
        - phi (double): phase shift
        - f (double): duty cycle. By default 0.5.
'''
def phaseGrCoeffs(n, phi, f=0.5):
    if n == 0:
        return 1 - f + f*np.exp(-1j*phi)
    else:
        return (np.exp(-1j*phi)-1)/(n*np.pi)*np.sin(n*np.pi*f)
        

# values for D3 - D1
dvals = np.linspace(-5e-2, 5e-2, 251)

freq = []
cont = []

for D in dvals:
    
    D3 = D + D1
    L3 = L - (L1 + D1 + D3)
    
    # fd = 1/Pd
    fd = abs(((f3-f2)*(L-L3)+(f1-f2)*L1-f2*(D1-D3))/L)
    freq.append(fd)
    
    B1 = phaseGrCoeffs(1, np.pi)
    Bm1 = phaseGrCoeffs(-1, np.pi)
    X2 = B1*np.conj(Bm1)
    
    delta1 = wvl*f1*L1/L*((f1-f2)*(L-L1)+(f3-f2)*L3-f2*(D3-D1))
    delta3 = wvl*f3*L3/L*((f3-f2)*(L-L3)+(f1-f2)*L1-f2*(D1-D3))
    
    # Ambiguity functions.
    # Only orders m=-1 and m=1 are different than 0
    X1 = 0
    X3 = 0
    for m in range(-1,1):
        Am = phaseGrCoeffs(m, np.pi/2)
        Am1 = phaseGrCoeffs(m+1, np.pi/2)
        X1 += Am*np.conj(Am1)*np.exp(1j*2*np.pi*(m+1/2)*delta1)
        X3 += Am*np.conj(Am1)*np.exp(1j*2*np.pi*(m+1/2)*delta3)
        
    cont.append(np.abs(X1)*np.abs(X2)*np.abs(X3))
    
freq = np.asarray(freq)
cont = np.asarray(cont)

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('D3-D1 [mm]')
ax1.set_ylabel('Contrast', color=color)
ax1.plot(dvals*1e3, cont, color=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color)
ax2.plot(dvals*1e3, freq*1e-3, color=color)

fig.tight_layout()
plt.show()

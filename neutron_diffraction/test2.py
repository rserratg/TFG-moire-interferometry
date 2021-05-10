# 3PGMI - theoretical frequency and contrast of moire fringes
# Using expression from Miao2016
# Using parameters from Sarenac2018
# Polychromatic beam: Maxwell-Boltzmann distribution

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats, constants

# OPTIONS
approx = True  # use n = +/- 1 only
store = False    # store data in json
datapath  = './contrast_data/PGMI3_sarenac2018_analytical.json'

# PARAMETERS

T = 40 # Maxwell distrib
sw = 500e-6

P = 2.4e-6
f1 = f2 = f3 = 1/P

L = 8.8     # Slit to detector
Ls2 = 4.75  # Slit 2nd grating
D1 = 4.6e-2

# D3 - D1
dvals = np.linspace(-25e-3, 25e-3, 201)

# Constants
m = constants.neutron_mass
hbar = constants.hbar
kb = constants.Boltzmann


#v = np.linspace(0,3000,3001)
wvlvals = np.linspace(0.10e-9, 10e-9, 100)
wvlweights = stats.maxwell.pdf(2*np.pi*hbar/m/wvlvals, scale=np.sqrt(kb*T/m))
wvlweights /= wvlweights.sum()

plt.plot(wvlvals*1e9, wvlweights)
plt.xlabel('wavelength [nm]')
plt.ylabel('weight')
plt.show()

# AUXILIARY FUNCTIONS

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


# CALCULATION

freq = []
cont = []

for D in dvals:

    print(D)

    if D == 0:
        freq.append(0)
        cont.append(0)
        continue
    
    L1 = Ls2 - D1
    D3 = D1 + D
    L3 = L - D3 - Ls2
    
    # fd = 1/Pd
    fd = abs(((f3-f2)*(L-L3)+(f1-f2)*L1-f2*(D1-D3))/L)
    freq.append(fd)
    
    c = 0
    
    for wvl, weight in zip(wvlvals, wvlweights):
    
        delta1 = wvl*f1/L*(f1*L1*(D1+D3+L3) - 2*f2*L1*(D3+L3) + f3*L1*L3)
        delta2 = wvl*f2/L*(f1*L1*(D3+L3) - 2*f2*(L1+D1)*(D3+L3) + f3*(L1+D1)*L3)
        delta3 = wvl*f3/L*(f1*L1*L3 - 2*f2*(L1+D1)*L3 + f3*(L1+D1+D3)*L3)
        
        # Ambiguity functions X1 and X3
        # Only orders m=-1 and m=0 are different than 0
        X1 = 0
        X3 = 0
        for m in range(-1,1):
            Am = phaseGrCoeffs(m, np.pi/2)
            Am1 = phaseGrCoeffs(m+1, np.pi/2)
            X1 += Am1*np.conj(Am)*np.exp(-1j*2*np.pi*(m+1/2)*delta1)
            X3 += Am1*np.conj(Am)*np.exp(-1j*2*np.pi*(m+1/2)*delta3)
            
        if approx:
            # X2 - approx
            B1 = phaseGrCoeffs(1, np.pi)
            Bm1 = phaseGrCoeffs(-1, np.pi)
            X2 = B1*np.conj(Bm1)
        else:
            # "Ambiguity function" X2
            # Only odd orders are different than 0
            X2 = 0
            for n in range(-13, 19, 2):
                Bn = phaseGrCoeffs(n, np.pi)
                Bn2 = phaseGrCoeffs(n-2, np.pi)
                X2 += Bn2*np.conj(Bn)*np.exp(-1j*2*np.pi*(n-1)*delta2)
        
        cwvl = X1*X2*X3
                
        # Source period
        Ps = P*L/D
        cwvl *= Ps/(sw*np.pi)*np.sin(sw*np.pi/Ps)
        
        c += cwvl * weight
    
    cont.append(2*np.abs(c))
    
freq = np.asarray(freq)
cont = np.asarray(cont)

# PLOT

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

if store:

    print('Storing results')
    data = {}
    data['dvals'] = dvals.tolist()
    data['contrast'] = cont.tolist()
    data['frequency'] = freq.tolist()
    with open(datapath, 'w') as fp:
        json.dump(data, fp)

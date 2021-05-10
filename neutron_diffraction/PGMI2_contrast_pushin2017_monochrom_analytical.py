# 2PGMI - theoretical frequency and contrast of moire fringes
# Parameters from Pushin et al 2017
# Monochromatic beam

import numpy as np
import matplotlib.pyplot as plt
import json

wvl = 0.44e-9
sw = 200e-6

P = 2.4e-6
f1 = f2 = 1/P
phi = 0.27*np.pi

L1 = 1.2
L = 2.99

Dvals = np.linspace(7e-3, 16e-3, 19)

# Fourier series coeffs for 50/50 phase grating
def coefs(n, phi):
    if n == 0:
        return 1/2 + 1/2*np.exp(-1j*phi)
    else:
        return (np.exp(-1j*phi) - 1)/(n*np.pi)*np.sin(n*np.pi/2)

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
    for m in range(-1,1):

        Am = coefs(m, phi)
        Am1 = coefs(m+1, phi)

        X1 += Am*np.conj(Am1)*np.exp(1j*2*np.pi*(m+1/2)*delta1)
        X2 += np.conj(Am)*Am1*np.exp(-1j*2*np.pi*(m+1/2)*delta2)

    # Average intensity transmission through gratings = 1
    c = np.abs(X1)*np.abs(X2)
    
    # Source period
    Ps = L/((f1-f2)*L2+f1*D)
    c *= Ps/(sw*np.pi)*np.sin(sw*np.pi/Ps)

    cont.append(c)

freq = np.asarray(freq)
cont = np.asarray(cont)

# Store results
data = {}
data['dvals'] = Dvals.tolist()
data['contrast'] = cont.tolist()
data['frequency'] = freq.tolist()

#with open('./contrast_data/PGMI2_pushin2017_monochrom_analytical.json', 'w') as fp:
#    json.dump(data, fp)


# Plot

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color)
ax1.plot(Dvals*1e3, cont*2, color=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color)
ax2.plot(Dvals*1e3, freq*1e-3, color=color)

fig.tight_layout()
plt.show()

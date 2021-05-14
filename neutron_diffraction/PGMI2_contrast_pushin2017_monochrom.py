# 2PGMI - Neutron - contrast
# Parameters from Pushin et al 2017
# Monochromatic beam

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
import json

# PARAMETERS    

# Sim
Nx = 2.5e4
Sx = 5e-3  # Set to wavefunction extension at camera (a bit more to avoid edge issues)
Nn = 100    # Number of neutrons per iteration
iters = 250 # Num of iterations. Total neutrons = iters*Nn

# Source
sw = 200e-6
wvl = 0.44e-9
theta = 0.3 * np.pi/180 # Maximum divergence for wavefunction's edge to be inside camera region

# Setup
L1 = 1.2
L = 2.99
dvals = np.linspace(7e-3, 16e-3, 19)

# Gratings
P = 2.4e-6
phi = 0.27*np.pi

# Plotting
xmin = -10e-3
xmax = 10e-3
numbins = int(1e4)

cont = []
freq = []

for D in dvals:

    print()
    print(f"D: {D}")
    print()
    
    L2 = L - L1 - D
    
    center = np.empty(numbins-1)
    hist = np.zeros(numbins-1)
    
    for ii in range(iters):
    
        print(ii+1)
        
        wave = NeutronWave(Nx, Sx, Nn=Nn, wvl=wvl)
        wave.slitSource(L1, sw, theta=theta, randPos=True)
        wave.rectPhaseGrating(P,phi)
        wave.propagate(D, pad=False)
        wave.rectPhaseGrating(P,phi)
        wave.propagate(L2, pad=False)
        
        center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
        hist += htemp
    
    P0 = P*L/D
    C, Pd, fit = contrast_fit(center, hist, P0, fitP = True)
   
    print()
    print(f"Contrast: {C}")
    print(f"Period: {Pd}")

    #plt.plot(center, hist)
    #plt.plot(center, fit, '--')
    #plt.show()
    
    cont.append(C)
    freq.append(1/Pd)
    
# Convert to numpy arrays
cont = np.asarray(cont)
freq = np.asarray(freq)
    
print('Storing results')
data = {}
data['dvals'] = dvals.tolist()
data['contrast'] = cont.tolist()
data['frequency'] = freq.tolist()
with open('./contrast_data/PGMI2_pushin2017_sim_test.json', 'w') as fp:
    json.dump(data, fp)

print('Plotting')

fig, ax1 = plt.subplots()
    
color1 = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color1)
#ax1.set_ylim(0, 1)
ax1.plot(dvals*1e3, cont, 'o', color=color1)

ax2 = ax1.twinx()

color2 = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color2)
ax2.plot(dvals*1e3, freq*1e-3, 'x', color=color2)

# Theoretical frequency
Fd = (1/P)*dvals/L
ax2.plot(dvals*1e3, Fd*1e-3, '-', color=color2)

fig.tight_layout()
plt.show()

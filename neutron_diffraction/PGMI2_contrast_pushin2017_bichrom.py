# 2PGMI - Neutron - contrast
# Parameters from Pushin et al 2017
# Bichromatic beam

# Nn = total number of neutrons
# For each wavelength, round(Nn*weight) neutrons are used

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
import json

# Parameters

# Sim 
Nx = 2.5e4
Sx = 5e-3
Nn = 250000 # Total number of neutrons
Nn_it = 250 # Maximum number of neutrons per iteration

# Source
sw = 200e-6
theta = 0.3 * np.pi/180 # Maximum divergence for wavefunction's edge to be inside camera region

wvlvals = np.array([0.22e-9, 0.44e-9])
wvlweights = np.array([1, 3.2])
wvlweights /= np.sum(wvlweights)

# Setup
L1 = 1.73
L = 3.52
dvals = np.linspace(7e-3, 16e-3, 19)

# Gratings 
P = 2.4e-6
phi = 0.27 * np.pi

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
    
    for wvl, weight in zip(wvlvals, wvlweights):
    
        print()
        print(f"Wvl: {wvl}")
        print()
        
        Nn_wvl = int(np.round(Nn*weight))
        
        while Nn_wvl > 0:
            
            print(f"Neutrons remaining: {Nn_wvl}")
            
            N = Nn_it if Nn_wvl >= Nn_it else Nn_wvl
            Nn_wvl -= N
            
            wave = NeutronWave(Nx, Sx, Nn=N, wvl=wvl)
            wave.slitSource(L1, sw, theta=theta, randPos=True)
            wave.rectPhaseGrating(P,phi)
            wave.propagate(D, pad=False)
            wave.rectPhaseGrating(P,phi)
            wave.propagate(L2, pad=False)
            
            center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
            hist += htemp
            
    P0 = P*L/D
    C, Pd, fit = contrast_fit(center, hist, P0, fitP=True)
    
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
with open('./contrast_data/PGMI2_pushin2017_bichrom_sim', 'w') as fp:
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
        

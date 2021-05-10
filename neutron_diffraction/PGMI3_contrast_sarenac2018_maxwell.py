# 3PGMI - Neutron - contrast
# Parameters from Sarenac et al 2018
# Polychromatic beam - Maxwell distribution

# Nn = total number of neutrons

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
from scipy import stats, constants
import json

# OPTIONS
plt = True
store = True
datapath = './contrast_data/PGMI3_sarenac2018_maxwell.json'

# PARAMETERS

# Sim 
Nx = 10e4
Sx = 10e-3
Nn = 250000 # Total number of neutrons

# Source
sw = 500e-6
theta = 0.3 * np.pi/180 # Maximum divergence for wavefunction's edge to be inside camera region
T = 40 

# Setup
L = 8.8
Ls2 = 4.75
D1 = 4.6e-2

# D3-D1
dvals = np.linspace(-25e-3, 25e-3, 51)

# Gratings 
P = 2.4e-6
phi1 = np.pi*0.5
phi2 = np.pi
phi3 = np.pi*0.5

# Plotting
xmin = -10e-3
xmax = 10e-3
numbins = int(1e4)

# Constants
m = constants.neutron_mass
hbar = constants.hbar
kb = constants.Boltzmann

cont = []
freq = []

for D in dvals:

    print()
    print(f"D: {D}")
    print()
    
    L1 = Ls2 - D1
    D3 = D1 + D
    L3 = L - D3 - Ls2
    
    center = np.empty(numbins-1)
    hist = np.zeros(numbins-1)
    
    for neutron in range(Nn):
    
        if neutron % 1000 == 0:
            print(f"  {neutron}")
            
        scale = np.sqrt(kb*T/m)
        v = stats.maxwell.rvs(scale=scale)
        wvl = 2*np.pi*hbar/m/v
        
        wave = NeutronWave(Nx, Sx, Nn=1, wvl=wvl)
        wave.slitSource(L1, sw, theta=theta, randPos=True)
        wave.rectPhaseGrating(P,phi1)
        wave.propagate_nopad(D1)
        wave.rectPhaseGrating(P,phi2)
        wave.propagate_nopad(D3)
        wave.rectPhaseGrating(P,phi3)
        wave.propagate_nopad(L3)
        
        center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
        hist += htemp
            
    P0 = abs(P*L/D)
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

# Store results in json file
if store:
    print('Storing results')
    data = {}
    data['dvals'] = dvals.tolist()
    data['contrast'] = cont.tolist()
    data['frequency'] = freq.tolist()
    with open(datapath, 'w') as fp:
        json.dump(data, fp)
    
# Plot
if plot:
    
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
        

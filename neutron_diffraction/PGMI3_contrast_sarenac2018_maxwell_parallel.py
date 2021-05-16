# 3PGMI - Neutron - contrast
# Parameters from Sarenac et al 2018
# Polychromatic beam - Maxwell distribution
# Parallelised at top level (Dvals)

# Nn = total number of neutrons

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
from scipy import stats, constants
from multiprocessing import Pool, cpu_count
import json

# OPTIONS
plot = False
store = True
datapath = './contrast_data/PGMI3_sarenac2018_maxwell_par.json'
numprocs = 9

# PARAMETERS

# Sim 
Nx = 20e4
Sx = 20e-3
Nn = 25000 # Total number of neutrons

# Source
sw = 500e-6
theta = 0.2 * np.pi/180 # Maximum divergence for wavefunction's edge to be inside camera region
T = 40 

# Setup
L = 8.8
Ls2 = 4.75
D1 = 4.6e-2

# D3-D1
dvals = np.linspace(-25e-3, 25e-3, 26)
dvals = dvals[np.abs(dvals) >= 0.5e-3]

# Gratings 
P = 2.4e-6
phi1 = np.pi*0.5
phi2 = np.pi
phi3 = np.pi*0.5

# Plotting
xmin = -20e-3
xmax = 20e-3
numbins = int(2e4)

# Constants
m = constants.neutron_mass
hbar = constants.hbar
kb = constants.Boltzmann

# Contrast for a given value of D
def contrast_D(D):

    print(f"Started D: {D*1e3} mm")
        
    L1 = Ls2 - D1
    D3 = D1 + D
    L3 = L - D3 - Ls2
    
    center = np.empty(numbins-1)
    hist = np.zeros(numbins-1)
    
    for neutron in range(Nn):
    
        if neutron % 1000 == 0:
            print(f"D = {D*1e3}mm -> {neutron} neutrons")
            
        scale = np.sqrt(kb*T/m)
        v = stats.maxwell.rvs(scale=scale)
        wvl = 2*np.pi*hbar/m/v
        
        wave = NeutronWave(Nx, Sx, Nn=1, wvl=wvl)
        wave.slitSource(L1, sw, theta=theta, randPos=True)
        wave.rectPhaseGrating(P,phi1)
        wave.propagate(D1, pad=False)
        wave.rectPhaseGrating(P,phi2)
        wave.propagate(D3, pad=False)
        wave.rectPhaseGrating(P,phi3)
        wave.propagate(L3, pad=False)
        
        center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
        hist += htemp
            
    P0 = abs(P*L/D)
    _, Pd, _ = contrast_fit(center, hist, P0, fitP=True)
    C, _, fit = contrast_fit(center, hist, Pd, fitP=False)
    
    #print()
    #print(f"Contrast: {C}")
    #print(f"Period: {Pd}")
    
    #plt.plot(center, hist)
    #plt.plot(center, fit, '--')
    #plt.show()
    
    return C, 1/Pd
    
# Main script
if __name__ == "__main__":

    print(f"Cpu count: {cpu_count()}")
    print(f"Processes used: {numprocs}")

    cont = []
    freq = []

    with Pool(numprocs) as pool:
        res = pool.map(contrast_D, dvals)
        cont, freq = zip(*res)

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
        

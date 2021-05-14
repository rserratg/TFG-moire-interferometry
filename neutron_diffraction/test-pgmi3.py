# PGMI3 - pattern - test - execution in parallel

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
from multiprocessing import Pool, cpu_count

# Sim
Nx = 20e4
Sx = 20e-3
Nit = 100

# Source
sw = 500e-6
Nn = 100
wvl = 0.5e-9
theta = 0.2*np.pi/180

# Setup
L = 8.8
Ls2 = 4.75
D1 = 4.6e-2
D = 1.5e-2 # D3-D1

L1 = Ls2 - D1
D3 = D1 + D
L3 = L - Ls2 - D3

# Gratings
P = 2.4e-6
phi1 = np.pi*0.5
phi2 = np.pi
phi3 = np.pi*0.5

# Plotting
xmin = -20e-3
xmax = 20e-3
numbins = int(2e4)

# Single iteration
def single_iteration(itnum):
    print(f"Iteration started: {itnum}")
    wave = NeutronWave(Nx, Sx, Nn=Nn, wvl=wvl)
    wave.slitSource(L1, sw, theta=theta, randPos=True)
    wave.rectPhaseGrating(P,phi1)
    wave.propagate(D1, pad=False)
    wave.rectPhaseGrating(P,phi2)
    wave.propagate(D3, pad=False)
    wave.rectPhaseGrating(P,phi3)
    wave.propagate(L3, pad=False)
    _, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
    print(f"Iteration finished: {itnum}")
    return htemp

# Main script
if __name__ == "__main__":
    
    print(f"Cpu count: {cpu_count()}")

    center = np.empty(numbins-1)
    hist = np.zeros(numbins-1)    

    # Run iterations in parallel
    with Pool(cpu_count()) as pool:
        res = pool.map(single_iteration, range(Nit))
        res = np.asanyarray(res)
        hist = res.sum(axis=0)/Nit

    # Reference field to get center
    wave = NeutronWave(Nx, Sx, Nn=1)
    center, _ = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)

    # Fit
    _, Pd, _ = contrast_fit(center, hist, abs(P*L/D), fitP=True)
    C, _, fit = contrast_fit(center, hist, Pd, fitP=False)

    # Print results
    print()
    print(f"Period: {abs(P*L/D)*1e3} mm")
    print(f"Period fit: {Pd}")
    print(f"Contrast: {C}")

    # Plot
    plt.plot(center*1e3, hist, '-')
    plt.plot(center*1e3, fit, '--', color='orange')
    plt.xlabel('x [mm]')
    plt.show()

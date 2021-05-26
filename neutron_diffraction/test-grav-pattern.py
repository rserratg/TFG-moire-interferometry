# PGMI3 - pattern - test - execution in parallel

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
from multiprocessing import Pool, cpu_count
from scipy.constants import g, m_n
from scipy.optimize import curve_fit

# Parallel
procs = 5

# Potential
alpha = 0.25*np.pi/180
F = g * m_n * np.sin(alpha)

# Sim
Nx = 20e4
Sx = 20e-3
Nit = 100

# Source
sw = 500e-6
Nn = 250
wvl = 0.5e-9
theta = 0.2*np.pi/180

# Setup
L = 8.8
Ls2 = 4.75
D1 = 1
D = 1e-2 # D3-D1

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

# Fit sine and return contrast, phase and fit
def contrast_fit_phase(x, u, P0):
    
    def fun(xx, a, b, c):
        return a + b*np.sin(2*np.pi*xx/P0 + c)

    def jac(xx, a, b, c):
        da = np.ones_like(xx)
        db = np.sin(2*np.pi*xx/P0 + c)
        dc = b*np.cos(2*np.pi*xx/P0 + c)
        return np.asanyarray([da, db, dc]).transpose()

    p0 = (np.mean(u), np.std(u), 0)
    popt, _ = curve_fit(fun, x, u, p0=p0, jac=jac)

    # Fit not reliable when using bounds
    #bmin = [0, 0, -2*np.pi]
    #bmax = [np.inf, np.inf, 2*np.pi]
    #popt, _ = curve_fit(fun, x, u, p0=p0, bounds=(bmin,bmax), jac=jac)

    A, B, phi = popt
   
    # If B<0 fix phase
    if B < 0:
        phi += np.pi
 
    phi = phi % (2*np.pi)
    C = abs(B/A)
    fit = fun(x, A, B, phi)
    return C, phi, fit

# Single iteration
def single_iteration(itnum):
    print(f"Iteration started: {itnum}")
    wave = NeutronWave(Nx, Sx, Nn=Nn, wvl=wvl)
    wave.slitSource(L1, sw, theta=theta, randPos=True)
    wave.rectPhaseGrating(P,phi1)
    wave.propagate_linear_potential(D1, F, pad=False)
    wave.rectPhaseGrating(P,phi2)
    wave.propagate_linear_potential(D3, F, pad=False)
    wave.rectPhaseGrating(P,phi3)
    wave.propagate_linear_potential(L3, F, pad=False)
    _, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
    print(f"Iteration finished: {itnum}")
    return htemp

# Main script
if __name__ == "__main__":
    
    print(f"Cpu count: {cpu_count()}")

    center = np.empty(numbins-1)
    hist = np.zeros(numbins-1)    

    # Run iterations in parallel
    with Pool(procs) as pool:
        res = pool.map(single_iteration, range(Nit))
        res = np.asanyarray(res)
        hist = res.sum(axis=0)/Nit

    # Reference field to get center
    wave = NeutronWave(Nx, Sx, Nn=1)
    center, _ = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)

    # Fit
    _, Pd, _ = contrast_fit(center, hist, abs(P*L/D), fitP=True)
    C, phase, fit = contrast_fit_phase(center, hist, Pd)

    # Print results
    print()
    print(f"Rotation: {alpha*180/np.pi} deg")
    print(f"Period: {abs(P*L/D)*1e3} mm")
    print(f"Period fit: {Pd}")
    print(f"Contrast: {C}")
    print(f"Phase: {phase} rad")

    # Plot
    plt.plot(center*1e3, hist, '-')
    plt.plot(center*1e3, fit, '--', color='orange')
    plt.xlabel('x [mm]')
    plt.show()

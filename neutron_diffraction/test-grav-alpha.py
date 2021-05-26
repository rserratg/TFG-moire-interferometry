# 3PGMI - Neutron - contrast
# Parameters from Sarenac et al 2018

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from neutronpckg.utils import contrast_fit
from scipy.optimize import curve_fit
from scipy.constants import g, m_n
from multiprocessing import Pool, cpu_count
import json

# PARAMETERS    

# Options
plot = True
numprocs = 7 # number of processes to run in parallel

# Sim
Nx = 20e4
Sx = 20e-3  # Set to wavefunction extension at camera (a bit more to avoid edge issues)
Nn = 100    # Number of neutrons per iteration
iters = 250 # Num of iterations. Total neutrons = iters*Nn

# Source
sw = 500e-6
wvl = 0.5e-9
theta = 0.2 * np.pi/180 # Maximum divergence for wavefunction's edge to be inside camera region

# Setup
L = 8.8
Ls2 = 4.75
D1 = 1
D = 1e-2

L1 = Ls2 - D1

# Potential
#alpha = angle of interf with respect to horizontal
avals = np.linspace(0, 10*np.pi/180, 41)
F = g*m_n

# Gratings
P = 2.4e-6
phi1 = np.pi*0.5
phi2 = np.pi
phi3 = np.pi*0.5

# Plotting
xmin = -20e-3
xmax = 20e-3
numbins = int(2e4)

# Fit sine and return contrast, phase and fitted function
# Period fixed at P0
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
    
    #bmin = [0, 0, -2*np.pi]
    #bmax = [np.inf, np.inf, 2*np.pi]
    #popt, _ = curve_fit(fun, x, u, p0=p0, bounds=(bmin,bmax), jac=jac)
    
    A, B, phi = popt

    if B < 0:
        phi += np.pi
    
    phi = phi%(2*np.pi)
    C = abs(B/A)
    fit = fun(x, A, B, phi)
    return C, phi, fit

# Contrast for a given value of alpha
def contrast_alpha(alpha):

    print(f"Started: {alpha} rad")    

    D3 = D1 + D
    L3 = L - Ls2 - D3

    center = np.empty(numbins-1)
    hist = np.zeros(numbins-1)

    for ii in range(iters):

        print(f"Iter: {ii} for alpha = {alpha} rad")

        wave = NeutronWave(Nx, Sx, Nn=Nn, wvl=wvl)
        wave.slitSource(L1, sw, theta=theta, randPos=True)
        wave.rectPhaseGrating(P,phi1)
        wave.propagate_linear_potential(D1, F*np.sin(alpha), pad=False)
        wave.rectPhaseGrating(P,phi2)
        wave.propagate_linear_potential(D3, F*np.sin(alpha), pad=False)
        wave.rectPhaseGrating(P,phi3)
        wave.propagate_linear_potential(L3, F*np.sin(alpha), pad=False)

        center, htemp = wave.hist_intensity(numbins, xlimits=(xmin,xmax), retcenter=True)
        hist += htemp

    P0 = abs(P*L/D)
    _, Pd, _ = contrast_fit(center, hist, P0, fitP=True)
    C, phase, fit = contrast_fit_phase(center, hist, Pd)

    print(f"Finished: {alpha} rad")

    return C, 1/Pd, phase

# Main script
if __name__ == "__main__":

    print(f"Cpu count: {cpu_count()}")

    cont = []
    freq = []
    phase = []

    with Pool(numprocs) as pool:
        res = pool.map(contrast_alpha, avals)
        cont, freq, phase = zip(*res)

    # Convert to numpy arrays
    cont = np.asarray(cont)
    freq = np.asarray(freq)
    phase = np.asarray(phase)

    # Print
    for a, c, f, p in zip(avals, cont, freq, phase):
        print()
        print(f"Rotation: {a} rad")
        print(f"Period: {1/f*1e3} mm")
        print(f"Contrast: {c}")
        print(f"Phase: {p} rad")

    # Plot

    if plot:

        print('Plotting')

        # avals from rad to deg
        avals *= 180/np.pi

        fig, ax1 = plt.subplots()
        
        color1 = 'tab:blue'
        ax1.set_xlabel('alpha [deg]')
        ax1.set_ylabel('Contrast', color=color1)
        #ax1.set_ylim(0, 1)
        ax1.plot(avals, cont, '-o', color=color1)

        ax2 = ax1.twinx()

        color2 = 'tab:red'
        
        #ax2.set_ylabel('Frequency [mm^-1]', color=color2)
        #ax2.plot(avals, freq*1e-3, 'x', color=color2)

        ax2.set_ylabel('Phase [rad]', color=color2)
        ax2.plot(avals, phase, '-x', color=color2)

        fig.tight_layout()
        plt.show()

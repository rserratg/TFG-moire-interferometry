# Script to plot and fit pattern
# Data generated with "PGMI3_gravity_pattern.py"

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg.utils import contrast_fit
from scipy.optimize import curve_fit
import pickle

# Options
datapath = "./contrast_data/PGMI3_gravity_pattern_2deg.data"

# Parameters
L = 8.8         # total distance from source to camera
D = 1e-2        # displacement of third grating w.r.t. echo plane (D3-D1)
P = 2.4e-6      # grating period

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
    popt, pcov = curve_fit(fun, x, u, p0=p0, jac=jac)
    
    print()
    print("Covariance matrix fit:")
    print(pcov)

    # Fit not reliable when using bounds
    #bmin = [0, 0, -2*np.pi]
    #bmax = [np.inf, np.inf, 2*np.pi]
    #popt, _ = curve_fit(fun, x, u, p0=p0, bounds=(bmin,bmax), jac=jac)

    A, B, phi = popt
    fit = fun(x, A, B, phi)
   
    # If B<0 fix phase
    if B < 0:
        phi += np.pi
 
    phi = phi % (2*np.pi)
    C = abs(B/A)
    return C, phi, fit
    

# Main script
if __name__ == "__main__":
    
    with open(datapath, "rb") as fp:
        data = pickle.load(fp)
        
    center = data["center"]     # sampling points in histogram
    hist = data["hist"]         # field intensity histogram
    
    # Fit
    _, Pd, _ = contrast_fit(center, hist, abs(P*L/D), fitP=True)
    C, phase, fit = contrast_fit_phase(center, hist, Pd)
    
    # Print results
    print()
    #print(f"Rotation: {alpha*180/np.pi} deg")
    print(f"Period: {abs(P*L/D)*1e3} mm")
    print(f"Period fit: {Pd}")
    print(f"Contrast: {C}")
    print(f"Phase: {phase} rad")
    
    # Plot
    plt.plot(center*1e3, hist, '-')
    plt.plot(center*1e3, fit, '--', color='orange')
    plt.xlabel('x [mm]')
    plt.show()

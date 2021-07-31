# Script to plot phase vs rotation
# Data from PGMI3_gravity_rotation.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy import constants
from scipy.stats import chisquare
import pickle

# OPTIONS
if False:
    datapath = "./contrast_data/PGMI3_gravity_rotation.data"
else:
    datapath = "./contrast_data/PGMI3_gravity_rotation_nopotential.data"

# --------------------

# PARSE DATA
with open(datapath, "rb") as fp:
    data = pickle.load(fp)
    
rotation = data["rotation"]
phase = data["phase"]
phase_err = data["phase_err"]

# PREPARE DATA FOR FIT
# manually unwrap phase for proper fit (fit of modded data does not perform well)
mod = -2*np.pi
loc = 6.6 * np.pi/180
phase[rotation > loc] += mod

# FIT

# Function: a + b*sin(alpha) mod 2pi
# B*sin(alpha) is the expected relative phase
# A is a correction for the absolute phase
def fun(alpha, a, b):
    return a + b*np.sin(alpha)
    
# Jacobian
def jac(alpha, a, b):
    da = np.ones_like(alpha)
    db = np.sin(alpha)
    return np.asanyarray([da, db]).transpose()
    
# Initial guess?
p0 = (phase[0], (phase[-1]-phase[0])/(np.sin(rotation[-1])-np.sin(rotation[0])))
popt, pcov = curve_fit(fun, rotation, phase, p0=p0, sigma=phase_err, absolute_sigma=True)
A, B = popt
perr = np.sqrt(np.diag(pcov))
Berr = perr[1]

print(f"Fit guess: {p0}")
print(f"Fit output: {A}, {B}")
print(f"Fit error: {perr}")

# Get fitted function for plotting
fit = fun(rotation, A, B)

# GRAVITY CALCULATION

# Analytical calculation of B
def B_calc(d12, d23, theta, m, g, k0):
    #hbar = 1.054e-34
    hbar = constants.hbar
    E0 = hbar**2*k0**2/2/m
    
    return k0*m*g/2/E0*(np.sin(theta)/2/np.cos(theta)**2*(d23**2 - d12**2) - d12*d23*np.tan(theta))
    
# Neutron parameters
#m = 1.67493e-27
m = constants.m_n
wvl = 5e-10

# Grating parameters
D1 = 1          # G1 to G2
D3 = 1.01       # G2 to G3
L3 = 3.04       # G3 to cam
P = 2.4e-6      # Grating period
theta = wvl/P   # Diffraction angle
Pd = 2.112e-3   # Period of pattern at camera

# Fit parameters
B_fit = B
B_err = Berr
#g_val = 9.80665
g_val = constants.g

B_val = B_calc(D1, D3, theta, m, g_val, 2*np.pi/wvl)

# Extra phase accounting for last propagation
Lsq = (D1+D3+L3)**2 - (D1+D3)**2
hbar = constants.hbar
v = 2*np.pi*hbar/m/wvl
extra = np.pi/Pd * Lsq/v**2 * g_val

# Extra phase accounting for last propagation (without potential)
extra_nopot = 2*np.pi/Pd * L3 * (D1+D3) * m**2*g_val / (hbar*2*np.pi/wvl)**2

# Extra phase due to gravity inside the gratings
# TODO: check this
h = constants.h
extra_inside = np.pi*wvl**2*m**2 / (P*h**2) * g_val * ((D1+D3)**2 - 2*D3**2)

# Theoretical line (using same A as fit)
thline = A + B_val * np.sin(rotation)

print()
print(f"Expected B: {B_val}")
print(f"Fit B: {B_fit}")
print(f"Fit uncertainty: {B_err}")
print(f"Relative uncertainty: {100*Berr/B_fit} %")
print(f"Relative error: {100*abs(1-B_fit/B_val)} %")
print()
print(f"Phase correction after G3: {extra}")
print(f"Phase correction after G3 (no potential): {extra_nopot}")
print(f"Phase correction between gratings: {extra_inside}")
print()
print("Fit g: %.3f"%(B_fit/B_calc(D1, D3, theta, m, 1, 2*np.pi/wvl)))
print("Relative Uncertainty: %.3f%%"%(100*abs(1-g_val/B_fit*B_calc(D1, D3, theta, m, 1, 2*np.pi/5e-10))))

# CHI-SQ TEST
print()
print("Chi-square test results (chisq, p-value):")
print(chisquare(phase, f_exp=fit))
    
# PLOT
# Phase (in rad ) vs rotation (in deg)

rotation = rotation*180/np.pi

fig = plt.figure()
gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3,1])
ax1, ax2 = gs.subplots(sharex=True)

ax1.plot(rotation, fit % (2*np.pi), color='tab:orange')
ax1.errorbar(rotation, phase % (2*np.pi), yerr=phase_err, fmt='.', color='tab:blue')
ax1.plot(rotation, thline % (2*np.pi), '--', color='tab:green')

ax2.errorbar(rotation, phase - fit, yerr=phase_err, fmt='.', color='tab:red')

ax1.set_ylabel("Phase [rad]")
ax2.set_ylabel("Residuals [rad]")
plt.xlabel("Rotation [deg]")
ax1.tick_params(axis="x", direction='in')
plt.tight_layout()
plt.show()

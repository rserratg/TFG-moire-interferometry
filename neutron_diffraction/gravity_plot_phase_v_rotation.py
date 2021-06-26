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

datapath = "./contrast_data/PGMI3_gravity_rotation.data"

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
d12 = 1
d23 = 1.01
P = 2.4e-6
theta = wvl/P 

# Fit parameters
B_fit = B
B_err = Berr
#g_val = 9.80665
g_val = constants.g

B_val = B_calc(d12, d23, theta, m, g_val, 2*np.pi/wvl)

# Extra phase accounting for last propagation
Lsq = 2.01**2 #  5.05**2 - 2.01**2
Pd = 2.112e-3
hbar = constants.hbar
v = 2*np.pi*hbar/m/wvl
extra = np.pi/Pd * Lsq/v**2 * g_val

# Theoretical line (using same A as fit)
thline = A + B_val * np.sin(rotation)

print()
print(f"Expected B: {B_val}")
print(f"Fit B: {B_fit}")
print(f"Fit uncertainty: {B_err}")
print(f"Relative uncertainty: {100*Berr/B_fit} %")
print(f"Relative error: {100*abs(1-B_fit/B_val)} %")
print(f"Phase correction: {extra}")
print()
print("Fit g: %.3f"%(B_fit/B_calc(d12, d23, theta, m, 1, 2*np.pi/wvl)))
print("Relative Uncertainty: %.3f%%"%(100*abs(1-g_val/B_fit*B_calc(d12, d23, theta, m, 1, 2*np.pi/5e-10))))

# CHI-SQ TEST
print()
print("Chi-square test results (chisq, p-value):")
print(chisquare(phase, f_exp=thline))
    
# PLOT
# Phase (in rad ) vs rotation (in deg)

rotation = rotation*180/np.pi

'''
fig1 = plt.figure(1)

# Plot data
frame1 = fig1.add_axes((.1,.3,.8,.6))
plt.plot(rotation, fit % (2*np.pi), color='tab:orange')
plt.errorbar(rotation, phase % (2*np.pi), yerr=phase_err, fmt='.', color='tab:blue')
plt.plot(rotation, thline % (2*np.pi), '--', color='tab:green')
frame1.set_ylabel("Phase [rad]")
frame1.set_xticklabels([]) # remove x-tick labels for the first frame

# Plot residuals (phase - theoretical values)
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.errorbar(rotation, phase - thline, yerr=phase_err, fmt='.', color='tab:red')
frame2.set_ylabel("Residuals [rad]")

plt.xlabel("Rotation [deg]")
plt.show()
'''

fig = plt.figure()
gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3,1])
ax1, ax2 = gs.subplots(sharex=True)

ax1.plot(rotation, fit % (2*np.pi), color='tab:orange')
ax1.errorbar(rotation, phase % (2*np.pi), yerr=phase_err, fmt='.', color='tab:blue')
ax1.plot(rotation, thline % (2*np.pi), '--', color='tab:green')

ax2.errorbar(rotation, phase - thline, yerr=phase_err, fmt='.', color='tab:red')

ax1.set_ylabel("Phase [rad]")
ax2.set_ylabel("Residuals [rad]")
plt.xlabel("Rotation [deg]")
ax1.tick_params(axis="x", direction='in')
plt.tight_layout()
plt.show()

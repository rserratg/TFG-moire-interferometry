# Script to calculate gravity value g
# From phase in plot vs rotation of the 3PGMI
# B is the relative phase obtained from the fit in gravity_plot_phase_v_rotation.py

import numpy as np

def phi(alpha, d12, d23):
    '''Returns the slope of a line '''
    theta = 5e-10/2.4e-6
    m = 1.67493e-27
    return A(d12, d23, theta) + B(d12, d23, theta, m, g)*np.sin(alpha)


def A(d12, d23, theta):
    return k0*(d12 - d23)*(1 - 1/np.cos(theta))

def B(d12, d23, theta, m, g, k0):
    hbar = 1.054e-34
    E0 = hbar**2*k0**2/2/m
    
    return k0*m*g/2/E0*(np.sin(theta)/2/np.cos(theta)**2*(d23**2 - d12**2) - d12*d23*np.tan(theta))

# Neutron parameters
m = 1.67493e-27
wvl = 5e-10

# Grating parameters
d12 = 1
d23 = 1.01
theta = 5e-10/2.4e-6 

# Fit parameters
B_fit = -41.26110146081525
B_err = 0
g_val = 9.80665

B_val = B(d12, d23, theta, m, g_val, 2*np.pi/5e-10)

print(f"Expected B: {B_val}")
print(f"Fit B: {B_fit}")
print(f"Fit uncertainty: {B_err}")
print(f"Relative error: {100*(1-B_fit/B_val)}")
print()
print("Fit g: %.3f"%(B_fit/B(d12, d23, theta, m, 1, 2*np.pi/5e-10)))
print("Relative Uncertainty: %.3f%%"%(100*abs(1-g_val/B_fit*B(d12, d23, theta, m, 1, 2*np.pi/5e-10))))

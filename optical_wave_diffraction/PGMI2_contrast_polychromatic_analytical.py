# 2PGMI - theoretical frequency and contrast of moire fringes
# Using expression from Miao 2016
# Polychromatic beam

# Wavelength distribution: 
# Gaussian with mean=1540nm and sigma=10nm

# To account for polychromaticity:
# Choose random value of wavelength in distribution
# Frequency is the same no matter the wavelength
# Since H0 = 1, contrast is simply the average over all H1(wvl)

import numpy as np
import matplotlib.pyplot as plt

# Setup params

P = 180e-6
f1 = f2 = 1/P

# L1 = d(point source, 1st grating)
L1 = 75e-3 + 32e-2
L = 1

Dvals = np.linspace(0, 200e-3, 401)
Dvals = Dvals[1:]

#########

# Wavelength distribution
#   - wvlvals contains wavelengths to be evaluated
#   - wvlweights contains weight of each wavelength.
#     If None, all values have equal weight (uniform distrib.)
        
# NIR laser normal distrib.
#wvlvals = np.random.normal(1550e-9, 20e-9, 1000)
#wvlweights = None

# NIR laser approximated spectrum
wvlvals = np.arange(1530, 1561, 2.5)*1e-9
wvlweights = np.array([0, 0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.65, 0.8, 1, 0.1, 0, 0])

#########

# Fourier series coeffs for pi/2 grating
def pi2coeffs(n):
    if n == 0:
        return 1/2 + 1/2*np.exp(-1j*np.pi/2)
    else:
        return (np.exp(-1j*np.pi/2)-1)/(n*np.pi)*np.sin(n*np.pi/2)

# Main script

freq = []
cont = []

# Calculate frequency and contrast for each value of D
for D in Dvals:

    print(f'D: {D}')

    L2 = L - L1 - D
    
    Pd = L/((f2-f1)*L1+f2*D)
    freq.append(1/Pd)
    
    # Contrast is a weighted average between contrast of all wavelengths
    cwvl = [] # contrast for each wavelength in wvlvals
    for wvl in wvlvals:
    
        delta1 = wvl/L*f1*L1*((f1-f2)*L2+f1*D)
        delta2 = wvl/L*f2*L2*((f2-f1)*L1+f2*D)
        
        X1 = 0
        X2 = 0
        for m in range(-1, 1):
    
            Am = pi2coeffs(m)
            Am1 = pi2coeffs(m+1)
            
            X1 += Am*np.conj(Am1)*np.exp(1j*2*np.pi*(m+1/2)*delta1)
            X2 += np.conj(Am)*Am1*np.exp(-1j*2*np.pi*(m+1/2)*delta2)
            
        cwvl.append(np.abs(X1)*np.abs(X2))
            
    cd = np.average(cwvl, weights=wvlweights) # contrast for d value
    cont.append(cd)
    
freq = np.asarray(freq)
cont = np.asarray(cont)

# Plot

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color)
ax1.plot(Dvals*1e3, cont, color=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color)
ax2.plot(Dvals*1e3, freq*1e-3, color=color)

fig.tight_layout()
plt.show()

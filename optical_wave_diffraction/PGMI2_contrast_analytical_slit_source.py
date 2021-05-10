'''
    PMGI2 - THEORETICAL CONTRAST
    
    Contrast plot to compare with fig 2b in Miao et. al. 2016
    (A universal moiré effect and application in X-ray phase-contrast imaging)
    
    - Extended source: slit (integral of H1 over slit distrib.) 
                       affects with the sine term in the contrast formula
    - Polychromatic source: white light. Spectrm matching approximately the spectrum shown
                            in the supplementary information. 
                            First peak at 450nm not considered to match with plot in paper.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Fourier series coeffs for pi/2 grating
def pi2coeffs(n):
    if n == 0:
        return 1/2 + 1/2*np.exp(-1j*np.pi/2)
    else:
        return (np.exp(-1j*np.pi/2)-1)/(n*np.pi)*np.sin(n*np.pi/2)
        
        
# Params universal moire
L1 = 10e-2
P = 14.3e-6
f1 = f2 = 1/P
Dvals = np.linspace(0.05e-3, 5e-3, 300)
L = 20e-2
sw = 0.44e-3 # slit width

wvlvals = np.linspace(350, 800, 501)
#wvlvals = np.linspace(450, 700, 25)
wvlweights = 2.5e6*skewnorm.pdf(wvlvals, 5, loc=500, scale=100)
#wvlweights += 26000*np.exp(-(wvlvals-450)**2/25**2)
wvlvals *= 1e-9

plt.plot(wvlvals*1e9, wvlweights, 'o-')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity [a.u.]')
plt.show()
#exit()

# Main script

freq = []
cont = []

# Calculate frequency and contrast for each value of D
for D in [1.1e-2]:

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
            
        # Note: X1 and X2 are real values, but the result can be either positive or negative
        # For appropiate results, the average should take into account this sign
        # For this reason the absolute value is taken at the end
        #c = np.abs(X1)*np.abs(X2)
        c = X1*X2
        
        # Source period
        Ps = L/((f1-f2)*L2+f1*D)
        c *= Ps/(sw*np.pi)*np.sin(sw*np.pi/Ps)
        
        cwvl.append(c)
            
    cd = np.average(cwvl, weights=wvlweights) # contrast for d value
    cont.append(np.abs(cd))
    
freq = np.asarray(freq)
cont = np.asarray(cont)

# Plot

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color)
ax1.plot(Dvals*1e3, cont*2, '-', color=color)
ax1.set_xlim([0,5])
ax1.set_ylim([0,1])

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color)
ax2.plot(Dvals*1e3, freq*1e-3, color=color)
ax2.set_ylim([0,2])

fig.tight_layout()
plt.show()
exit()

import json
data = {}
data['dvals'] = Dvals.tolist()
data['contrast'] = cont.tolist()
data['frequency'] = freq.tolist()
with open('./plots/Tests/PGMI2/Contrast_data/PGMI2_miao2016_analytical.json', 'w') as fp:
    json.dump(data, fp)

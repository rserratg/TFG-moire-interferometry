# 2PGMI - Contrast of polychromatic, extended source
# Miao 2016

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import normalizedIntensity, intensity, contrast_FT
from scipy.stats import skewnorm
import json

# Numerical parameters
N = 8e4
Sx = 8e-2

# Beam parameters
wvlvals = np.linspace(450, 700, 25)
wvlweights = 2.5e6*skewnorm.pdf(wvlvals, 5, loc=500, scale=100)
wvlvals *= 1e-9

theta_max = 8*np.pi/180
theta_num = 151

# Propagation
L1 = 10e-2
L = 20e-2

Dvals = np.linspace(0.1e-3, 5e-3, 50)

# Grating settings
P = 14.3e-6
phi = np.pi/2

# Get x-space (independent of wavelength)
ref = OptWave(N, Sx, wvl=None)
x = ref.x
d = ref.d

# Run sim
print('Running simulation...')

freq = []
cont = []

for D in Dvals:

    print()
    print(f"D: {D}")
    print()
    
    I = np.zeros(int(N))

    for wvl, weight in zip(wvlvals, wvlweights):

        #print()
        print(f"wavelength: {wvl}")
        #print()

        Iwvl = np.zeros(int(N))

        for theta in np.linspace(0, theta_max, theta_num):

            #print(f"theta: {theta}")

            wave = OptWave(N,Sx,wvl)
            wave.planeWave(theta=theta)
            wave.rectAperture(0.44e-3)
            wave._conv_propagation(L1, "AS", pad=False)
            wave.rectPhaseGrating(P, phi)
            wave._conv_propagation(D, "AS", pad=False)
            wave.rectPhaseGrating(P, phi)
            wave._conv_propagation(L-D-L1, "AS", pad=False)
            
            Ith = intensity(wave.U)
            Iwvl += Ith
            
            # Use symmetry to calculate negative angles
            if theta > 0:
                Iwvl[1:] += np.flip(Ith)[:-1]
            
        I += Iwvl * weight
        
    Pmoire = P*L/D
    
    # Method 1: period and contrast with FT
    #C, fd = contrast_FT(d, I, 1/Pmoire)
    
    # Method 2: period with fit, contrast with second fit (fixed period)
    _, Pd, _ = contrast_period(x, I, Pmoire, xlim=(-18e-3, 18e-3))
    C = contrast(x, I, Pd, xlim=(-18e-3, 18e-3))
    fd = 1/Pd
    
    freq.append(fd)
    cont.append(C)
    
    print()
    print(f"Contrast: {C}")
    print(f"Period: {1/fd*1e3} mm")
    
# Convert to numpy array
freq = np.asarray(freq)
cont = np.asarray(cont)

# Store data
print('Storing data')
data = {}
data['dvals'] = Dvals.tolist()
data['contrast'] = cont.tolist()
data['frequency'] = freq.tolist()
with open('./plots/Tests/PGMI2/Contrast_data/PGMI2_miao2016_sim.json', 'w') as fp:
    json.dump(data, fp)

# Plot
print('Plotting')

fig, ax1 = plt.subplots()

color1 = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color1)
ax1.plot(Dvals*1e3, cont, 'o', color=color1)

ax2 = ax1.twinx()

color2 = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color2)
ax2.plot(Dvals*1e3, freq*1e-3, 'x', color=color2)

# Theoretical frequency
Fd = (1/P)*Dvals/L
ax2.plot(Dvals*1e3, Fd*1e-3, '-', color=color2)

fig.tight_layout()
#plt.show()


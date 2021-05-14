# PGMI2 - contrast of moire fringes, calculated with the Fourier Transform method
# Simulation 1: contrast of a single pattern
# Simulation 2: plot contrast and frequency vs grating separation D

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity, contrast_FT
import json    
    
########################################################

# GENERAL PARAMETERS

numsim = 1

# Options sim 2
plot = False
store = False
datapath = "./Contrast_data/PGMI2_contrast.json"

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Grating parameters
P = 180e-6

# Lens
f = -75e-3


# Sim 1: contrast from a single output pattern
def contrast_single():
    
    # Numerical parameters
    N = 1e6
    S = 100e-3
    
    # Setup
    L0 = 11e-2
    L1 = 98e-2
    D = 8.8e-2
    L = 2 + 75e-3
    L2 = L - (-f + L1 + D)
    
    if L2 < 0:
        print('Invalid parameters: negative propagation')
        return
        
    print('Calculating output intensity...')
    
    wave = OptWave(N, S, wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, np.pi/2)
    wave.angular_spectrum(D)
    wave.rectPhaseGrating(P, np.pi/2)
    wave.angular_spectrum(L2)
    
    # Get results
    x = wave.x
    d = wave.d
    I = intensity(wave.U)
    
    print('Calculating contrast...')
    
    fmoire = (1/P)*D/L          # expected freq of moire fringes
    ftalbot = (1/P)*(-f+L1)/L   # freq of Talbot fringes from first grating
    C, fd = contrast_FT(d, I, fmoire, None, plotft=True)
    
    # Show results
    print()
    print(f'Contrast = {C}')
    print(f'Calculated Moire period = {1/fd*1e3} mm')
    print(f'Expected Moire period = {1/fmoire*1e3} mm')
    print(f'Expected Talbot period = {1/ftalbot*1e3} mm')
    
    # Plot output intensity
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [a.u.]')
    plt.show()
     
    
# Sim 2: plot contrast and frequency vs D
def contrast_vs_D():
    
    # Numerical parameters
    N = 1e5
    S = 60e-3
    
    # Setup
    L0 = 11e-2
    L1 = 98e-2
    L = 2 + 75e-3
    
    dvals = np.linspace(1e-2, 11e-2, 201)
    
    print('Initial propagation')
    
    wave = OptWave(N, S, wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, np.pi/2)
    
    # Store intermediate results
    x = wave.x
    d = wave.d
    u = wave.U
    
    # Arrays to store results
    C = []
    F = []
    
    for D in dvals:
        
        wave.U = u
        
        print(f'D: {D}')
        L2 = L - (-f + L1 + D)
        
        wave.angular_spectrum(D)
        wave.rectPhaseGrating(P, np.pi/2)
        wave.angular_spectrum(L2)
        
        I = intensity(wave.U)
        
        fmoire = (1/P)*D/L
        ftalbot = (1/P)*(-f+L1)/L
        c, fd = contrast_FT(d, I, fmoire, ftalbot)
        
        C.append(c)
        F.append(fd)
        
    # Conver to numpy array
    C = np.asarray(C)
    F = np.asarray(F)

    # Store results
    if store:
        
        print("Storing data")

        data = {}
        data['dvals'] = dvals.tolist()
        data['contrast'] = C.tolist()
        data['frequency'] = F.tolist()
        with open(datapath, 'w') as fp:
            json.dump(data, fp)

    # Plot
    if plot:
    
        print('Plotting...')
    
        fig, ax1 = plt.subplots()
    
        color1 = 'tab:blue'
        ax1.set_xlabel('D [mm]')
        ax1.set_ylabel('Contrast', color=color1)
        #ax1.set_ylim(0, 1)
        ax1.plot(dvals*1e3, C, 'o', color=color1)
    
        ax2 = ax1.twinx()
    
        color2 = 'tab:red'
        ax2.set_ylabel('Frequency [mm^-1]', color=color2)
        ax2.plot(dvals*1e3, F*1e-3, 'x', color=color2)
    
        # Theoretical frequency
        Fd = (1/P)*dvals/L
        ax2.plot(dvals*1e3, Fd*1e-3, '-', color=color2)
    
        fig.tight_layout()
        plt.show()

####################################

# Main script
if numsim == 1:
    contrast_single()
elif numsim == 2:
    contrast_vs_D()
else:
    print('Invalid sim number')
    exit()

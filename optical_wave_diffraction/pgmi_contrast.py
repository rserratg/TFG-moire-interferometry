# SIMULATION 2PGMI / 3PMGI - CONTRAST FIGURES

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity, contrast, contrast_period, rebin

numsim = 1

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Grating settings
P = 180e-6

# Lens
f = -75e-3


# Sim 1: contrast from a single output pattern in 2pgmi
def contrast_single_pattern():
    
    N = 5e5
    S = 100e-3 # field size
    
    L0 = 11e-2
    L1 = 32e-2
    D = 30e-3
    L = 1
    L2 = L - (-f + L1 + D)
    
    if L2 < 0:
        print('Invalid parameters: negative propagation')
        return      
    
    print('Calculating output pattern...')
    
    # Output pattern
    wave = OptWave(N,S,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, np.pi/2)
    wave.angular_spectrum(D)
    wave.rectPhaseGrating(P, np.pi/2)
    wave.angular_spectrum(L2)
    
    print('Calculating reference field...')
    
    # Reference pattern
    ref = OptWave(N,S,wvl)
    ref.gaussianBeam(w0, z0=z0)
    ref.angular_spectrum(L0)
    ref.lens(f)
    ref.angular_spectrum(L + f)
    
    # Get results
    x = wave.x
    I = intensity(wave.U)
    Iref = intensity(ref.U)
    Is = I/Iref
    
    '''
    print('Rebinning...')
    
    x, Is = rebin(x, Is, 500, avg=True)
    print('Rebin px:', x[1]-x[0])
    '''
    
    print('Fitting...')
    
    # Calculate contrast from known period
    Pd = P*L/D
    c, Pf, sdPf, fit = contrast_period(x, Is, Pd, xlim=(-10e-3,10e-3), retfit = True)
    print('Expected period:', Pd*1e3, 'mm')
    print('Contrast:', c)
    print('Fitted period:', Pf*1e3, 'mm')
    print('Period standard deviation:', sdPf*1e3, 'mm')
   
    print('Plotting...')
    
    plt.plot(x*1e3, I)
    plt.plot(x*1e3, fit*Iref, '--')
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arb. units]')
    plt.show()
    

# Sim 2: 2PGMI - contrast and freq/period vs D
def contrast_2pgmi():

    N = 5e5
    S = 60e-3
    
    L0 = 11e-2
    L1 = 32e-2
    L = 1
    
    dvals = np.linspace(1e-3, 200e-3, 200)
    
    # Parametres provisionals
    N = 1e5
    S = 20e-3
    wvl = 0.55e-6
    w0 = 0.44e-3
    z0 = 0
    f = -5e-3
    P = 14.4e-6
    L = 20e-2
    L1 = 10e-2
    dvals = np.linspace(0.05e-3, 5e-3, 100)
    
    print('Calculating reference field...')
    
    ref = OptWave(N,S,wvl)
    ref.gaussianBeam(w0, z0=z0)
    ref.angular_spectrum(L0)
    ref.lens(f)
    ref.angular_spectrum(L + f)
    
    print('Initial propagation...')
    
    # Field until first grating
    wave = OptWave(N, S, wvl)
    wave.gaussianBeam(w0, z0=z0)
    #wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, np.pi/2)
    
    x = wave.x
    u = wave.U
    Iref = intensity(ref.U)
    
    C = []
    Pd = []
    sdPd = []
    
    for D in dvals:
        wave.U = u
        
        print('D:', D)
        L2 = L - (-f + L1 + D)
        
        wave.angular_spectrum(D)
        wave.rectPhaseGrating(P, np.pi/2)
        wave.angular_spectrum(L2)
        
        I = intensity(wave.U)
        Is = I/Iref
        
        Pd0 = P*L/D # initial guess for Pd
        c, Pd_d, sdPd_d = contrast_period(x, Is, Pd0, xlim=(-10e-3, 10e-3))
        
        C.append(c)
        Pd.append(Pd_d)
        sdPd.append(sdPd_d)
        
        print('c: ', c)
        print('P: ', Pd_d)
        print()
        
    # Convert to numpy array
    C = np.asarray(C)
    Pd = np.asarray(Pd)
    sdPd = np.asarray(sdPd)
       
        
    print('Plotting...')
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('D [mm]')
    ax1.set_ylabel('Contrast', color=color)
    ax1.set_ylim(0, 1)
    ax1.plot(dvals*1e3, C, 'o-', color=color)
    
    ax2 = ax1.twinx()
    
    #color = 'tab:red'
    #ax2.set_ylabel('Period [mm]', color=color)
    #ax2.errorbar(dvals*1e3, Pd*1e3, yerr=sdPd*1e3, color=color, fmt='x')
    
    color = 'tab:red'
    ax2.set_ylabel('Frequency [mm^-1]', color=color)
    ax2.plot(dvals*1e3, 1/Pd*1e-3, 'x-', color=color)
    
    fig.tight_layout()
    plt.show()
    
    
if numsim == 1:
    contrast_single_pattern()
elif numsim == 2:
    contrast_2pgmi()
else:
    print('Invalid sim number')
    exit()

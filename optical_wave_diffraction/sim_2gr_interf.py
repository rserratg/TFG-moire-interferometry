# SIMULATION - 2 grating interferometer
# Simulations to compare with experimental results

'''
    - Gaussian beam
        - wvl = 1550 nm
        - beam waist = 1.6mm
        - waist distance = 7.26 mm
    - Propagate 40 mm
    - Diverging lens
        - f = -25 mm
    - Propagate 150 mm
    - 2 x phase grating
        - Phase = pi/2
        - Period = 180 um
    - Propagate 150 mm
    
    1. Beam propagation for a few centimeters after the diverging lens
    2. Talbot carpet for 100mm after a single pi/2 grating (& pi)
    3. Image for two pi/2 gratings with a separation of 20 mm
    4. Cross section of talbot carpet after a single slit at a number of distances
    
    Same scenario for non-diverging beam (same system without the lens)
    Rauyleigh range of beam ~ 5m (so beam can be considered as non-diverging)
    5. Beam propagation
    6. Talbot carpet for 100mm after a single pi/2 (& pi)
    7. Image for two pi/2 gratings with a separation of 20 mm
    
    8. Carpet after two pi/2 gratings
    
'''

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import normalizedIntensity, intensity

# COMMON SIMULATION SETTINGS

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L1 = 40e-3
L2 = 150e-3
D = 20e-3
L3 = 150e-3

# Grating settings
P = 180e-6

# Lens
f = -25e-3

# Number of simulation to run (see description on top of the file)
numsim = 8


# SIMULATIONS

# Sim 1: beam propagation after lens
def beamprop():
    print('Simulation 1')
    
    # Parameters
    N = 1e5 + 1
    L = 50e-3
    zmax = 500e-3
    znum = 501
    
    print('Performing initial propagation...')

    # Initial propagation (until lens)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L1)
    wave.lens(f)
    
    print('Storing results...')
    
    # Store intermediate result
    x = wave.x
    u = wave.U
   
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    # zmax is not calculated, only used for plotting purposes
    for z in zticks[:-1]:
        print('z: ', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = normalizedIntensity(wave.U)
        I.append(Iz)
        
    print('Plotting...')
        
    # Convert list to numpy array
    I = np.asanyarray(I)
    
    # Prepare x axis for plotting
    x = np.concatenate((x, [-x[0]]))
    
    # Plot I(x,z)
    plt.pcolormesh(x*1e3, zticks*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    plt.show()
    
# Sim 2: talbot carpet
def carpet():
    
    print('Simulation 2')

    # Parameters
    N = 1e5 + 1
    L = 50e-3
    zmax = 100e-3
    znum = 601
    phi = np.pi/2
    
    print('Performing initial propagation...')

    # Initial propagation (untils grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L1)
    wave.lens(f)
    wave.angular_spectrum(L2)
    wave.rectPhaseGrating(P, phi)
    
    print('Storing results...')
    
    # Store intermediate result
    x = wave.x
    u = wave.U
   
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    #zticks = np.linspace(0.1, 0.07, 301)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    # zmax is not calculated, only used for plotting purposes
    for z in zticks[:-1]:
        print('z: ', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = normalizedIntensity(wave.U)
        I.append(Iz)
        
    # Convert list to numpy array
    I = np.asanyarray(I)
    
    # Prepare x axis for plotting
    x = np.concatenate((x, [-x[0]]))
    
    print('Plotting...')
    
    # Plot I(x,z)
    plt.pcolormesh(x*1e3, zticks*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    #plt.xlim(-0.5, 0.5) # - 500 um to 500 um
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    #plt.show()
    
    print('Saving...')
    plt.savefig('Sim2_pi2_high_res.png', dpi=800)
    

# Sim 3: image after system
def pattern():
    
    print('Simulation 3')
    
    # Parameters
    N = 3e6 + 1
    L = 50e-3
    phi = np.pi/2

    print('Running simulation')

    # Run sim
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)  
    wave.angular_spectrum(L1)
    wave.lens(f)
    wave.angular_spectrum(L2)
    wave.rectPhaseGrating(P, phi)
    wave.angular_spectrum(D)
    wave.rectPhaseGrating(P, phi)
    wave.angular_spectrum(L3)
    
    # Get results
    x = wave.x
    I = normalizedIntensity(wave.U)
    
    print('Plotting...')
    
    # Plot
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.show()    
    

# Sim 4: cross section of talbot carpet
def cross_section():
    
    print('Simulation 4')
    
    # Parameters
    N = 3e6 + 1
    L = 50e-3
    phi = np.pi/2
    
    # zvals = [0.0055, 0.0073, 0.0083, 0.011, 0.0854, 0.0905, 0.0915, 0.0975] # pi gr
    zvals = [0.0111, 0.0140, 0.0190, 0.0237, 0.0742, 0.0790, 0.0852, 0.0974] # pi/2 gr
    
    print('Performing initial propagation...')
    
    # Initial propagation (until grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L1)
    wave.lens(f)
    wave.angular_spectrum(L2)
    wave.rectPhaseGrating(P, phi)
    
    print('Storing results...')
    
    # Store intermediate results
    x = wave.x
    u = wave.U
    
    print('Running simulation...')
    
    # Run sim and show each plot
    for z in zvals:
        print('z: ', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = normalizedIntensity(wave.U)
        
        plt.plot(x*1e3, Iz)
        plt.xlabel('x [mm]')
        plt.ylabel('Intensity [arbitrary units]')
        plt.xlim(-0.5, 0.5)
        plt.show()
        
# Sim 5: beam propagation (no lens)
def beamprop_nolens():
    print('Simulation 5')
    
    # Parameters
    N = 1e5 + 1
    L = 10e-3
    zmax = 500e-3
    znum = 501
    
    print('Performing initial propagation...')

    # Initial propagation
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L1)
    
    print('Storing results...')
    
    # Store intermediate result
    x = wave.x
    u = wave.U
   
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    # zmax is not calculated, only used for plotting purposes
    for z in zticks[:-1]:
        print('z: ', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = normalizedIntensity(wave.U)
        I.append(Iz)
        
    print('Plotting...')
        
    # Convert list to numpy array
    I = np.asanyarray(I)
    
    # Prepare x axis for plotting
    x = np.concatenate((x, [-x[0]]))
    
    # Plot I(x,z)
    plt.pcolormesh(x*1e3, zticks*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    plt.show()


# Sim 6: talbot carpet (no lens)
def carpet_nolens():
    
    print('Simulation 6')

    # Parameters
    N = 2e4
    L = 20e-3
    zmax = 100e-3
    znum = 501
    phi = np.pi/2
    
    print('Performing initial propagation...')

    # Initial propagation (untils grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L1+L2)
    wave.rectPhaseGrating(P, phi)
    
    print('Storing results...')
    
    # Store intermediate result
    x = wave.x
    u = wave.U
   
    # Prepare sim
    #zticks = np.linspace(0, zmax, znum)
    zticks = np.linspace(0, 50e-3, 201)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    # zmax is not calculated, only used for plotting purposes
    for z in zticks[:-1]:
        print('z: ', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = normalizedIntensity(wave.U)
        I.append(Iz)
        
    # Convert list to numpy array
    I = np.asanyarray(I)
    
    # Prepare x axis for plotting
    x = np.concatenate((x, [-x[0]]))
    
    print('Plotting...')
    
    # Plot I(x,z)
    plt.pcolormesh(x*1e3, zticks*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    plt.xlim(-0.5, 0.5)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    #plt.show()
    
    print('Saving...')
    plt.savefig('Sim6_pi2_high_res.png', dpi=200)


# Sim 7: image after system (no lens)
def pattern_nolens():
    
    print('Simulation 7')
    
    # Parameters
    N = 1e6 + 1
    L = 20e-3
    phi = np.pi/2

    print('Running simulation')

    # Run sim
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)  
    wave.angular_spectrum(L1+L2)
    wave.rectPhaseGrating(P, phi)
    wave.angular_spectrum(D)
    wave.rectPhaseGrating(P, phi, x0=P)
    wave.angular_spectrum(L3)
    
    # Get results
    x = wave.x
    I = normalizedIntensity(wave.U)
    
    print('Plotting...')
    
    # Plot
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.show() 
    

# Sim 8: carpet after system
def carpet_2gr():
    
    print('Simulation 8')
    
    # Parameters
    N = 1e5
    L = 50e-3
    phi = np.pi/2
    zmax = 100e-3
    znum = 501
    
    zt = 2*P**2/wvl
    D = 38.25e-3
    
    print('Performing initial propagation...')
    
    # Initial propagation (until second grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L1)
    wave.lens(f)
    wave.angular_spectrum(L2)
    wave.rectPhaseGrating(P, phi)
    wave.angular_spectrum(D)
    wave.rectPhaseGrating(P, phi)
    
    print('Storing results...')
    
    # Store intermediate results
    x = wave.x
    u = wave.U
    
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []

    print('Running simulation')

    # Run sim
    for z in zticks[:-1]:
        print('z: ', z)
        wave.U = u
        wave.angular_spectrum(z)
        #Iz = normalizedIntensity(wave.U)
        Iz= intensity(wave.U)
        I.append(Iz)
        
    # Convert list to numpy array
    I = np.asanyarray(I)
    
    # Prepare x axis for plotting
    x = np.concatenate((x, [-x[0]]))
    
    print('Plotting...')
    
    # Plot I(x,z)
    plt.pcolormesh(x*1e3, zticks*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    #plt.xlim(-5, 5)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    
    print('Saving...')
    
    plt.savefig('Sim8_pi2', dpi=800)


# MAIN SCRIPT
if numsim == 1:
    beamprop()
elif numsim == 2:
    carpet()
elif numsim == 3:
    pattern()
elif numsim == 4:
    cross_section()
elif numsim == 5:
    beamprop_nolens()
elif numsim == 6:
    carpet_nolens()
elif numsim == 7:
    pattern_nolens()
elif numsim == 8:
    carpet_2gr()
else:
    print('Incorrect sim number')
    exit()

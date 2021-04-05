# Three phase-grating moire interferometer
# Simulations to compare with experimental results

'''
    - Source to lens: 11 cm
    - Lens f = -75 mm
    - Lens to 1st gr: 32 cm
    - 1st gr to 2nd gr: 4 cm
    - 2nd gr to 3rd gr: 7 cm
    - 3rd gr to camera: 48 cm
    
    Assuming same source as 2gr interf:
        - Gaussian beam
        - wvl = 1550 nm
        - beam waist = 1.6mm
        - waist distance = 7.26 mm
        
    - Gratings:
        - phase = pi/2 (side ones), pi (middle one)
        - period = 180 um
        
    1. Pattern after system
    2. Carpet after 2nd grating
        "The second pi grating makes a virtual images of the first pi/2 grating"
    3. Pattern after 2nd grating
    4. Carpet of virtual images
       Carpets of patterns at L2 = D, changing D
'''

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import normalizedIntensity, intensity, contrast

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L0 = 11e-2
L1 = 32e-2
D1 = 4e-2
D2 = 8e-2
L2 = 48e-2

# Grating settings
P = 180e-6
phi1 = np.pi/2
phi2 = np.pi
phi3 = np.pi/2

# Lens
f = -75e-3

# Sim
numsim = 1


def pattern():
    
    print('Sim 1: pattern after system')
    
    # Parameters
    N = 1e6
    L = 60e-3
    
    D1 = 20e-2
    D2 = 28e-2
    
    print('Running simulation')
    
    # Run sim
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(D1)
    wave.rectPhaseGrating(P, np.pi/2)
    wave.angular_spectrum(D2)
    wave.rectPhaseGrating(P, phi3)
    wave.angular_spectrum(L2)
    
    # Get results
    x = wave.x
    I = normalizedIntensity(wave.U)
    
    print('Plotting...')
    
    # Plot
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.show()


def carpet_2gr():
    
    print('Sim 2: carpet after 2nd grating')
    
    # Parameters
    N = 1e5
    L = 50e-3
    zmax = 15e-2
    znum = 301
    
    D1 = 4e-2
    
    print('Performing initial propagation...')
    
    # Initial propagation (until second grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(D1)
    wave.rectPhaseGrating(P, phi2)
    
    print('Storing results...')
    
    # Store intermediate results
    x = wave.x
    u = wave.U
    
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    for z in zticks[:-1]:
        print('z: ', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = intensity(wave.U)
        #Iz = np.angle(wave.U)
        I.append(Iz)
        
    # Convert list to numpy array
    I = np.asanyarray(I)
    
    # Prepare x axis for plotting
    x = np.concatenate((x, [-x[0]]))
    
    print('Plotting...')
    
    # Plot I(x,z)
    plt.pcolormesh(x*1e3, zticks*1e2, I)
    plt.xlabel('x [mm]')
    plt.ylabel('z [cm]')
    plt.xlim(-5,5)
    #plt.ylim(2, 8)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    
    print('Saving...')
    
    plt.savefig('Sim2', dpi=800)

def pattern_2gr():
    
    print('Sim 3: pattern after 2nd grating')
    
    # Parameters
    N = 1e6
    L = 60e-3
    D1 = 30.7e-2
    D2 = 30.7e-2
    
    print('Running simulation')
    
    # Run sim
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(D1)
    wave.rectPhaseGrating(P, phi2)
    wave.angular_spectrum(D2)
    
    # Get results
    x = wave.x
    I = normalizedIntensity(wave.U)
    
    print('Plotting...')
    
    # Plot
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.xlim(-2.5, 2.5)
    plt.show()
    
def carpet_vi():
    
    print('Sim 4: carpet of virtual images')
    
    # Parameters
    N = 1e5
    L = 50e-3
    zmax = 50e-2
    znum = 501
    
    print('Performing initial propagation...')
    
    # Initial propagation (until second grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi1)
    
    print('Storing results...')
    
    # Store intermediate results
    x = wave.x
    u = wave.U
    
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    for D in zticks[:-1]:
        print('D: ', D)
        wave.U = u
        wave.angular_spectrum(D)
        wave.rectPhaseGrating(P, phi2)
        wave.angular_spectrum(D)
        Iz = intensity(wave.U)
        I.append(Iz)
        
    # Convert list to numpy array
    I = np.asanyarray(I)
    
    # Prepare x axis for plotting
    x = np.concatenate((x, [-x[0]]))
    
    print('Plotting...')
    
    # Plot I(x,z)
    plt.pcolormesh(x*1e3, zticks*1e2, I)
    plt.xlabel('x [mm]')
    plt.ylabel('D [cm]')
    plt.xlim(-2.5,2.5)
    #plt.ylim(2, 8)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    
    print('Saving...')
    
    plt.savefig('Sim2', dpi=800)


if numsim == 1:
    pattern()
elif numsim==2:
    carpet_2gr()
elif numsim==3:
    pattern_2gr()
elif numsim==4:
    carpet_vi()
else:
    print('Incorrect sim number')
    exit()

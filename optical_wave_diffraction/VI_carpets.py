'''
    Carpets after virtual images of the 3 PGMI
    
    1. Carpet after virtual image. pi/2 + pi setup.
    2. Carpet after a single pi/2 grating placed at virtual image plane
    3. Carpet after 3PGMI
    4. Carpet after 2PGMI (pi/2 + pi/2 setup).
       First grating placed at virtual image plane.
       Second grating placed at same plane as last grating in sim 3.
    5. Phase pattern at zero-contrast plane after virtual image.
    6. Phase pattern at zero-contrast plane after a single pi/2 grating.
'''

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity

numsim = 5

# Beam parameters
wvl = 1.55e-6
w0 = 1.6e-3
z0 = 7.26e-3

# Propagation
L0 = 11e-2
L1 = 32e-2

D1 = 4e-2
D3 = 6e-2
L3 = 26e-3

L2 = D1 # distance from 2nd gr to vi

# Grating settings
P = 180e-6
phi1 = np.pi/2
phi2 = np.pi
phi3 = np.pi/2
zt = 2*P**2/wvl

# Lens
f = -75e-3

def carpet_vi():
    
    print("Simulation 1: carpet after VI")
    
    N = 1e5
    L = 50e-3
    
    zmax = D1
    znum = 401
    
    print('Initial propagation...')
    
    # Initial propagation (until virtual image)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(D1)
    wave.rectPhaseGrating(P, phi2)
    wave.angular_spectrum(L2)
    
    print('Storing results...')
    
    x = wave.x
    u = wave.U
    
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    for z in zticks[:-1]:
        print('z:', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = intensity(wave.U)
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
    plt.xlim(-10,10)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    
    print('Saving...')
    
    plt.savefig('Sim_VI_1.png', dpi=400)


def carpet_single():

    print("Simulation 2: carpet after single grating")
    
    N = 1e5
    L = 50e-3
    
    zmax = D1
    znum = 401
    
    print('Initial propagation...')
    
    # Initial propagation until grating at virtual image plane
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1 + D1 + L2)
    wave.rectPhaseGrating(P, phi1)
    
    print('Storing results...')
    
    x = wave.x
    u = wave.U
    
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    for z in zticks[:-1]:
        print('z:', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = intensity(wave.U)
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
    plt.xlim(-10,10)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    
    print('Saving...')
    
    plt.savefig('Sim_VI_2.png', dpi=400)
    

def carpet_3pgmi():

    print("Simulation 3: carpet after 2PGMI")
    
    N = 1e5
    L = 50e-3
    
    zmax = 50e-2
    znum = 251
    
    print('Initial propagation...')
    
    # Initial propagation (until third grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(D1)
    wave.rectPhaseGrating(P, phi2)
    wave.angular_spectrum(D3)
    wave.rectPhaseGrating(P, phi3)
    
    print('Storing results...')
    
    x = wave.x
    u = wave.U
    
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    for z in zticks[:-1]:
        print('z:', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = intensity(wave.U)
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
    #plt.xlim(-5,5)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    
    print('Saving...')
    
    plt.savefig('Sim_VI_3.png', dpi=400)
    
    
def carpet_2pgmi():
    
    print("Simulation 4: carpet after 2PGMI")
    
    N = 1e5
    L = 50e-3
    
    zmax = 50e-2
    znum = 251
    
    print('Initial propagation...')
    
    # Initial propagation (until second grating)
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1 + D1 + L2)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(D3-L2)
    wave.rectPhaseGrating(P, phi3)
    
    print('Storing results...')
    
    x = wave.x
    u = wave.U
    
    # Prepare sim
    zticks = np.linspace(0, zmax, znum)
    I = []
    
    print('Running simulation...')
    
    # Run sim
    for z in zticks[:-1]:
        print('z:', z)
        wave.U = u
        wave.angular_spectrum(z)
        Iz = intensity(wave.U)
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
    #plt.xlim(-5,5)
    clb = plt.colorbar()
    clb.set_label('Intensity [arbitrary units]')
    plt.tight_layout()
    
    print('Saving...')
    
    plt.savefig('Sim_VI_4.png', dpi=400)
    

# d = 4cm -> low-contrast plane at L3 = 26mm
def pattern_vi():
    
    print('Sim 5: pattern after virtual image')
    
    N = 1e6
    L = 60e-3
    
    print('Running simulation...')
    
    # Run sim
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(D1)
    wave.rectPhaseGrating(P, phi2)
    wave.angular_spectrum(L2)
    
    # Get results
    x = wave.x
    I = np.angle(wave.U)
    
    print('Plotting...')
    
    # Plot
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Phase [rad]')
    plt.xlim(-5,5)
    plt.show()
    
    
# For d = 4cm, first zero-contrast plane at L3 = 21.75mm
def pattern_single():
    
    print('Sim 6: pattern after single grating')
    
    N = 1e6
    L = 60e-3
    
    print('Running simulation...')
    
    # Run sim
    wave = OptWave(N,L,wvl)
    wave.gaussianBeam(w0, z0=z0)
    wave.angular_spectrum(L0)
    wave.lens(f)
    wave.angular_spectrum(L1 + D1 + L2)
    wave.rectPhaseGrating(P, phi1)
    wave.angular_spectrum(L3)
    
    # Get results
    x = wave.x
    I = np.angle(wave.U)
    
    print('Plotting...')
    
    # Plot
    plt.plot(x*1e3, I)
    plt.xlabel('x [mm]')
    plt.ylabel('Phase [rad]')
    plt.xlim(-5,5)
    plt.show()


# Main script
if numsim == 1:
    carpet_vi()
elif numsim == 2:
    carpet_single()
elif numsim == 3:
    carpet_3pgmi()
elif numsim == 4:
    carpet_2pgmi()
elif numsim == 5:
    pattern_vi()
elif numsim == 6:
    pattern_single()
else:
    print('Incorrect sim number')
    exit()

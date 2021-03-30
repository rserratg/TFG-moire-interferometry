# Talbot interferometer
# Phase object imaging
# Setup from Ibarra 1992: "Talbot interferometry: a new geometry"

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg.utils import intensity
from scipy.optimize import differential_evolution

# Simulation settings

# General settings
N = 5e5
L = 10e-2
wvl = 0.6328e-6

# Gratings
P = 3.175e-4
zt = 2*P**2/wvl # 31.86 cm

# Imaging system
f = 25e-2 # focal length of imaging lens
D = 0.05e-2 # width of spatial filter (slit)

numsim = 6


# Sim 1: trapezoidal phase object imaging
def trapezoidobj():

    print("Simulation 1 - trapezoid")
    
    phi = np.pi
    a = 1e-2
    b = 1.1e-2
    
    wave = OptWave(N,L,wvl)
    wave.planeWave()
    wave.rectAmplitudeGrating(P, x0=0)
    wave.angular_spectrum(zt/4)
    wave.trapezoidPhaseObject(phi, a, b)
    wave.angular_spectrum(zt/4)
    wave.rectAmplitudeGrating(P)
    wave.angular_spectrum(2*f - zt/4)
    wave.lens(f)
    wave.angular_spectrum(f)
    wave.rectAperture(D, x0=-1e-3)
    wave.angular_spectrum(f)
    
    x = wave.x
    I = intensity(wave.U)
    
    plt.plot(x,I)
    plt.xlabel('x [m]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.show()
    
    # Peak integral
    #p1 = I[(x>-0.014) & (x<-0.008)].sum()
    #p2 = I[(x>0.008) & (x<0.014)].sum()

    #print(p1)
    #print(p2)


# Sim 2: lens imaging
# T(x) = exp(-1j * 2pi/wvl * x**2/2f)
def lensobj():

    print("Simulation 2 - lens")
    
    '''
        Lens test 
        f' = (1/n)*12.54 m
        2n = number of fringes
        Fringes are only seen if no filter is applied!
    '''
    fobj = 12.54/4
    
    wave = OptWave(N,L,wvl)
    wave.planeWave()
    wave.rectAmplitudeGrating(P, x0=0)
    wave.angular_spectrum(zt/4)
    wave.lens(fobj)
    wave.angular_spectrum(zt/4)
    wave.rectAmplitudeGrating(P)
    wave.angular_spectrum(2*f - zt/4)
    wave.lens(f)
    wave.angular_spectrum(f)
    #wave.rectAperture(D)
    #wave.angular_spectrum(f)
    
    x = wave.x
    I = intensity(wave.U)
    
    plt.plot(x,I)
    plt.xlabel('x [m]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.show()
    

# Sim 3: constant gradient (prism)
# T(x) = exp(-1j * phi * x)
# No imaging system
def prismobj():

    print("Simulation 3 - constant gradient (no imaging)")
    
    # For a displacement of 1 period -> phi ~ 40000
    phi = 39579.59/2
    
    wave = OptWave(N, L, wvl)
    wave.planeWave()
    wave.rectAmplitudeGrating(P, x0=0)
    wave.angular_spectrum(zt/4)
    
    wave.U *= np.exp(-1j * wave.x * phi)
    
    wave.angular_spectrum(zt/4)
    wave.rectAmplitudeGrating(P)
    
    wave.angular_spectrum(zt/4)
    
    #wave.angular_spectrum(2*f - zt/4)
    #wave.lens(f)
    #wave.angular_spectrum(f)
    #wave.rectAperture(D, x0=-0.0375e-2)
    #wave.angular_spectrum(f)
    
    x = wave.x
    I = intensity(wave.U)
    
    print('Fitting...')
    
    
    # Fit to rectangular wave pattern
    # Pattern period is expected to be P/2
    # Obtain amplitude, shift and duty cycle
    
    # Function to fit
    def fun(xx, *p):
        A, x0, ff = p
        t1 = np.cos(2*np.pi*(xx-x0)*2/P)
        t2 = np.cos(np.pi * ff)
        g = A*np.ones(len(xx))
        g[t1 < t2] = 0
        return g
        
    # Cannot use curve_fit because it relies on gradient methods
    # Our function has steps, and so curve_fit fails
    
    xaux = x[np.abs(x) < 30e-3]
    Iaux = I[np.abs(x) < 30e-3]
    
    sqdiff = lambda p : np.sum((fun(xaux,*p) - Iaux)**2)
    bounds = [[0., 2.], [0., P], [0.,1.]]
    res = differential_evolution(sqdiff, bounds)
    A, x0, ff = res.x
    Ifit = fun(x, A, x0, ff)
    
    print('Duty cycle:', ff)
    print(res.x)
    
    plt.plot(x*1e3,I)
    plt.plot(x*1e3, Ifit, '--')
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.xlim(-0.5, 0.5)
    plt.show()
    

# Sim 4: plot fit amplitude and duty cycle w.r.t. prism phase gradient
def prism_duty_cycle_study():
    
    print('Simulation 4 - amplitude/duty cycle vs. constant gradient')
    
    N = 1e5
    phivals = np.linspace(0, 40000, 401)
    
    print('Initial propagation...')
    
    # Initial propagation (until object)
    wave = OptWave(N,L,wvl)
    wave.planeWave()
    wave.rectAmplitudeGrating(P, x0=0)
    wave.angular_spectrum(zt/4)
    
    # Store result
    u = wave.U
    x = wave.x
    amplitudes = []
    duty_cycles = []
    
    print('Running simulation...')
    
    for phi in phivals:
        
        print('phi =', phi)
        
        # avoid overwriting u!
        wave.U = u*np.exp(-1j * wave.x * phi)
        
        wave.angular_spectrum(zt/4)
        wave.rectAmplitudeGrating(P)
        wave.angular_spectrum(zt/4)
        
        Iphi = intensity(wave.U)
        
        # Fit
        
        def fun(xx, *p):
            A, x0, ff = p
            t1 = np.cos(2*np.pi*(xx-x0)*2/P)
            t2 = np.cos(np.pi * ff)
            g = A*np.ones(len(xx))
            g[t1 < t2] = 0
            return g
        
        xaux = x[np.abs(x) < 30e-3]
        Iaux = Iphi[np.abs(x) < 30e-3]
        
        sqdiff = lambda p : np.sum((fun(xaux,*p) - Iaux)**2)
        bounds = [[0., 2.], [0., P], [0.,1.]]
        res = differential_evolution(sqdiff, bounds)
        A, x0, ff = res.x
        
        amplitudes.append(A)
        duty_cycles.append(ff)
        
    # Convert to numpy arrays
    amplitudes = np.asarray(amplitudes)
    duty_cycles = np.asarray(duty_cycles)
    
    print('Plotting...')
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('phi [rad/m]')
    ax1.set_ylabel('Amplitude [arb. units]', color='tab:blue')
    ax1.plot(phivals, amplitudes, color='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Duty cycle', color='tab:orange')
    ax2.plot(phivals, duty_cycles, color='tab:orange')
    plt.show()


# Sim 5: measure constant phase gradient based on pattern duty cycle
# Run fitting function multiple times and average duty cycle (since it is stochastic)
def prism_phase_measurement():

    print("Simulation 5 - constant gradient measurement")
    
    phi = 15000
    
    print('Propagating...')
    
    wave = OptWave(N, L, wvl)
    wave.planeWave()
    wave.rectAmplitudeGrating(P, x0=0)
    wave.angular_spectrum(zt/4)
    
    wave.U *= np.exp(-1j * wave.x * phi)
    
    wave.angular_spectrum(zt/4)
    wave.rectAmplitudeGrating(P)
    
    wave.angular_spectrum(zt/4)
    
    x = wave.x
    I = intensity(wave.U)
    
    print('Fitting...')
    
    def fun(xx, *p):
        A, x0, ff = p
        t1 = np.cos(2*np.pi*(xx-x0)*2/P)
        t2 = np.cos(np.pi * ff)
        g = A*np.ones(len(xx))
        g[t1 < t2] = 0
        return g
        
    xaux = x[np.abs(x) < 30e-3]
    Iaux = I[np.abs(x) < 30e-3]
    
    numfit = 10
    duty_cycles = []
    
    for n in range(numfit):
    
        print(n)
    
        sqdiff = lambda p : np.sum((fun(xaux,*p) - Iaux)**2)
        bounds = [[0., 2.], [0., P], [0.,1.]]
        res = differential_evolution(sqdiff, bounds)
        _, _, ff = res.x
        
        duty_cycles.append(ff)
    
    duty_cycles = np.asarray(duty_cycles)
    
    dc_avg = np.mean(duty_cycles)
    dc_std = np.std(duty_cycles)
    
    print("Mean:", dc_avg)
    print("Std:", dc_std)
    
    phi = 4*np.pi*P/(zt*wvl)*dc_avg
    
    print('Phi:', phi)
    
    
# Sim 6: custom shape
def customobj():

    print("Simulation 6 - custom object")
    
    wave = OptWave(N,L,wvl)
    wave.planeWave()
    wave.rectAmplitudeGrating(P, x0=0)
    wave.angular_spectrum(zt/4)
    
    #wave.trapezoidPhaseObject(phi, a, b)
        
    grad = 5000
    a = 2e-2
    b = 1e-2
    
    # Prism
    G = grad*(wave.x + a)
    G[wave.x < -a] = 0
    G[wave.x > a] = 2*grad*a 
    
    # Trapezoid
    #G = grad*(np.abs(wave.x) + a)
    #G[np.abs(wave.x) > a] = 2*grad*a
    #G[np.abs(wave.x) < b] = grad*(a+b)
    
    wave.U *= np.exp(-1j* G)
    
    plt.plot(wave.x, G)
    plt.show()
    
    wave.angular_spectrum(zt/4)
    wave.rectAmplitudeGrating(P)
    
    #wave.angular_spectrum(zt/4)
    
    wave.angular_spectrum(2*f - zt/4)
    wave.lens(f)
    wave.angular_spectrum(f)    
    #wave.rectAperture(D)
    #wave.angular_spectrum(f)
    
    x = wave.x
    I = intensity(wave.U)
    
    plt.plot(x*1e3,I)
    plt.xlabel('x [mm]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.show()
    
# Main script
if numsim == 1:
    trapezoidobj()
elif numsim == 2:
    lensobj()
elif numsim == 3:
    prismobj()
elif numsim == 4:
    prism_duty_cycle_study()
elif numsim == 5:
    prism_phase_measurement()
elif numsim == 6:
    customobj()
else:
    print("Incorrect sim number")
    exit()


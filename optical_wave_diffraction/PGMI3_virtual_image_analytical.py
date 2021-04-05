# 3-phase grating moiré interferometer
# Analysis of analytical expression for "virtual/Fourier image" from Miao 2016.

import numpy as np
import matplotlib.pyplot as plt

# Fourier series coefficients for pi/2 phase grating with 50% duty cycle
def pi2GratingCoef(m):
    if m == 0:
        return 1/2 + 1/2 * np.exp(-1j*np.pi/2)
    else:
        a = np.exp(-1j*np.pi/2) - 1
        b = m*np.pi
        return a*np.sin(b/2)/b
        
# COMMON PARAMETERS

# Simulation to run
# 1: pattern of virtual image at echo plane
# 2: carpet of virtual images at echo plane vs D
numsim = 1

# Beam 
wvl = 1.55e-6
k = 2*np.pi/wvl

# Pi/2 gratings
P = 180e-6
f1 = f2 = 1/P

# Pi grating - fourier series coeffs with 50% duty cycle (B1 = B_-1)
B1 = -2/np.pi

# Propagation from point source to first grating
L1 = 75e-3 + 32e-3


# Sim1: pattern of virtual image at echo plane
def pattern():
    
    # Numerical parameters
    L = 5e-3
    N = 2e5
    y = np.linspace(-L/2, L/2, N)
    
    # Distance between the gratings
    D = 21e-2
    
    L2 = D*f1/(2*f2-f1)
    L = L1 + D + L2
    
    # Fourier series orders
    mmax = 50
    
    # Run sim
    
    Ve = 0
    
    # Plot amplitude of each m term in the sum
    #plt.figure()
    
    for m in range(-mmax, mmax+1):
        
        Am = pi2GratingCoef(m)
        Am1 = pi2GratingCoef(m+1)
        
        phi0m = 2*np.pi*m*f1*(L1/L)*y 
        phi0n = 2*np.pi*f2*((L1+D)/L)*y
        phi0 = phi0m + phi0n
        
        phi1m = (2*np.pi*m*f1)**2*(L1/L)
        phi1n = (2*np.pi*f2)**2*(L2/L)
        phi1mn = (2*np.pi*m*f1*L1/L - 2*np.pi*f2*L2/L)**2
        phi1 = - L/(2*k) * (phi1m + phi1n - phi1mn)
        
        Vem = Am*B1 + Am1*B1*np.exp(1j*2*np.pi*(f1-2*f2)*y)
        Vem *= np.exp(1j * (phi0 + phi1))
        
        #plt.plot(y*1e3, np.real(Vem))
        
        Ve += Vem
        
    #plt.xlabel('y[mm]')
    #plt.ylabel('Amplitude [arb. units]')
    #plt.show()

    I = np.abs(Ve)**2
    
    plt.plot(y*1e3, I)
    plt.xlabel('y [mm]')
    plt.ylabel('Intensity')
    plt.xlim(-2.5, 2.5)
    plt.show()
    

# Sim2: carpet of virtual images at 'echo' plane vs D
def carpet():
    
    # Numerical parameters
    L = 5e-3
    N = 5e3
    y = np.linspace(-L/2, L/2, N)

    # Distance between gratings
    Dvals = np.linspace(0, 50e-2, 501)
    
    # Series orders 
    mmax = 50

    I = []
    
    # Run sim
    for D in Dvals:
        
        print('D:', D*1e2, 'cm')
        
        L2 = D*f1/(2*f2-f1)
        L = L1 + D + L2
        
        Ve = 0
        
        for m in range(-mmax, mmax+1):
        
            Am = pi2GratigCoef(m)
            Am1 = pi2GratingCoef(m+1)
            
            phi0m = 2*np.pi*m*f1*(L1/L)*y 
            phi0n = 2*np.pi*f2*((L1+D)/L)*y
            phi0 = phi0m + phi0n
            
            phi1m = (2*np.pi*m*f1)**2*(L1/L)
            phi1n = (2*np.pi*f2)**2*(L2/L)
            phi1mn = (2*np.pi*m*f1*L1/L - 2*np.pi*f2*L2/L)**2
            phi1 = - L/(2*k) * (phi1m + phi1n - phi1mn)
            
            Vem = Am*B1 + Am1*B1*np.exp(1j*2*np.pi*(f1-2*f2)*y)
            Vem *= np.exp(1j * (phi0 + phi1))
            
            Ve += Vem
            
        Id = np.abs(Ve)**2
        I.append(Id)
        
    # Convert to numpy array
    I = np.asanyarray(I)
    
    # Plot
    plt.colormesh(y*1e3, Dvals*1e2, I, shading='nearest')
    plt.xlabel('y [mm]')
    plt.ylabel('D [cm]')
    plt.colorbar()
    plt.show()


# Main script
if numsim == 1:
    pattern()
elif numsim == 2:
    carpet()
else:
    print('Invalid simulation number')
    exit()

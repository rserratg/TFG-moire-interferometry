import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


####################
#                  #
#   OptWave class  #
#                  #
####################


'''
    Class OptWave represents a 1D optical wave
    
    Properties:
        - d: x-space sampling interval
        - x: x-space sampling points
        - wvl: wavelength
        - U: wave function (complex) amplitude at sampling points
'''
class OptWave:

    '''
        Default constructor
        
        Parameters:
            - N: number of grid points
            - L: size of x-space
            - lamb: wavelength
            
        Post: 
            Initialize class properties according to input parameters.
            U initialized to zero.
    '''
    def __init__(self, N, L, lamb):
        self.d = L/N
        self.x = np.arange(-N//2, N//2)*self.d
        self.wvl = lamb
        self.U = np.zeros(N, dtype=np.complex128)
        
    
    '''
        1D Fourier Transform
        
        Parameters:
            - g: x-space amplitudes
            - delta: sampling interval
            
        Post:
            Returns freq-space amplitudes (DFT of g)
    '''
    def ft(g, delta):
        G = np.fft.fftshift(np.fft.fft(np.fft.fftshift(g)))
        return G*delta
        
        
    '''
        1D Inverse Fourier Transform
        
        Parameters:
            - G: x-space amplitudes
            - delta_f: frequency spacing
            
        Post:
            Returns x-space amplitudes (IDFT of G)
            
        Notes:
            Numpy normalizes the ifft by 1/N. 
    '''
    def ift(G, delta_f):
        g = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G)))
        return g*len(g)*delta_f
        
    '''
        Uniform plane wave
        
        Parameters:
            - A: amplitude of the wave (A=1 by default)
        
        Post:
            Returns plane wave of amplitude 1
            (Output array dtype is complex)
        
    '''
    def planeWave(self, A=1):
        self.U.fill(A)
    
        
    '''
        Rectangular Aperture
        t = rect(x/D)
        
        Parameters:
            - D: aperture width
            
        Post:
            Applies rectangular aperture mask to input plane wave.
            Updates wave.
    '''
    def rectAperture(self, D):
        self.U = np.where(np.abs(self.x) <= D/2, self.U, 0)
        
        
    '''
        Propagation of optical waves based on the Fresnel approximation
        Angular spectrum method (convolution theorem)
        (1D)
        
        Parameters:
            - z: propagation distance
            
        Post:
            - self.U = wave function after propagation
            - self.d, self.x are NOT modified
            
        Notes:
            - Direct application of convolution theorem results in dout = din, xout = xin  
              To define custom d, must rewrite the integral adding scaling parameter.
            - Best suited for only short distances
            - N >= wvl*z/(d**2)
            - d <= wvl*z/(D1+D2)
            
            - Transfer function not working properly
              Results differ between both versions (calculated by hand vs dft)
    '''
    def fresnel_ang_spec(self, z):
        N = len(self.U)
        k = 2*np.pi/self.wvl
        
        df = 1/(self.d*N)
        fx = np.arange(-N//2, N//2)*df;
        fsq = fx**2
        
        # Transfer function
        H = np.sqrt(self.wvl*z)*np.exp(1j*np.pi/4)*np.exp(-1j*np.pi*self.wvl*z*fsq)        
        #H = OptWave.ft(np.exp(1j*k/(2*z)*self.x**2), self.d)
        A = np.exp(1j*k*z)/(1j*self.wvl*z)
        
        self.U = A*OptWave.ift(H*OptWave.ft(self.U,self.d), df)
        
            
    '''
        Plane wave of unit amplitude
        Rectangular aperture
        Fresnel propagation
        (1D)
        
        Parameters:
            - z: propagation distance from aperture
            - L: aperture width
    '''
    def planeRectFresnelSolution(self, z, L):
        A = np.sqrt(self.wvl*z/2)
        k = 2*np.pi/self.wvl

        #substitutions
        alpha1 = (self.x + L/2)/A
        alpha2 = (self.x - L/2)/A
        
        # Fresnel sine and cosine integrals
        sa1, ca1 = sp.fresnel(alpha1)
        sa2, ca2 = sp.fresnel(alpha2)
        
        # Observation-plane field
        B = -A*np.exp(1j*k*z)/(1j*self.wvl*z)
        Uout = B*((ca2-ca1) + 1j*(sa2-sa1))
        return Uout

  
##########################      
#                        #
#      Main script       #
#                        #
##########################

# Example: 
# Plane wave (unit amplitude) incident on rectangular aperture
# Fresnel propagation via integral

#Sim parameters
N = 1024 # number of grid points
L = 1e-2 # total size of the grid [m]
wvl = 1e-6 # optical wavelength [m]
D = 2e-3 # diamater of the aperture [m]
z = 1 # propagation distance

# Sim computation
wave = OptWave(N,L,wvl)
wave.planeWave()
wave.rectAperture(D)
wave.fresnel_ang_spec(z)

# Get results
xout = wave.x
Uout = wave.U
Uan = wave.planeRectFresnelSolution(z,D)

# Plot results

# Notes on numpy: 
# Magnitude: np.abs(U)
# Phase: np.angle(U)

plt.plot(xout, np.abs(Uout), "-")
plt.plot(xout, np.abs(Uan), "--")
plt.show()

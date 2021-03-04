# Wavefunction initializers for the OptWave class

import numpy as np

'''
    Helper (Mixin) class for the OptWave class.
    
    Contains methods related to generation of wavefunctions.
'''

class MixinWave:

    '''
        Uniform plane wave
        
        self.U = A*exp(1j * k * self.x*sin(theta))
        
        Parameters:
            - A (double): amplitude of the wave (A=1 by default)
            - theta (double): angle in radians (theta=0 by default)        
        Post:
            self.U is a plane wave of amplitude A with an angle theta
            (Output array dtype is complex)
        
    '''
    def planeWave(self, A=1, theta=0):
        k = 2 * np.pi / self.wvl
        self.U = A * np.exp(1j*k*self.x*np.sin(theta))

    '''
        Gaussian beam
        
        Parameters:
            - w0 (double): minimum beam width 
            - x0 (double): position of center (x0=0 by default)
            - z0 (double): position w.r.t. beam waist (z0=0 by default)
                           Positive for divergent beam (right of beam waist)
                           Negative for divergent beam (left of beam waist)
            - A (double): maximum amplitude (A=1 by default)
            - theta (double): angle in radians (theta=0 by default)
            
        Post:
            self.U is a gaussian beam with the given parameters
            
        Note:
            - Expression for 1D gaussian beam, consistent with 1D Fresnel propagation.
              Amplitude and Gouy phase differ from 2D expression.
    '''        
    def gaussianBeam(self, w0, x0=0, z0=0, A=1, theta=0):
        k = 2 * np.pi / self.wvl
        
        # Rayleigh distance
        zr = k  * w0**2 / 2
        
        # Gouy phase
        gouy = np.arctan2(z0, zr)/2
        
        w = w0 * np.sqrt(1 + (z0/zr)**2)
        if z0 == 0:
            R = 1e10 # At beam waist, R = infinity
        else:
            R = z0 * (1 + (zr/z0)**2)
        amplitude = A * np.sqrt(w0/w) * np.exp(-(self.x-x0)**2 / w**2)
        phase1 = np.exp(1j * k * ((self.x-x0)*np.sin(theta))) # rotation
        phase2 = np.exp(1j * (k*z0 - gouy + k*(self.x-x0)**2 / (2*R)))
        
        self.U = amplitude * phase1 * phase2
        
    

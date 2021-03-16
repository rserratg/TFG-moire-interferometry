# Optical elements for the OptWave class

import numpy as np

'''
    Helper (Mixin) class for the OptWave class.
    
    Contains the methods related to optical elements.
'''

class MixinElem:

    '''
        Rectangular Aperture
        t = rect(x/D)
        
        Parameters:
            - D: aperture width
            - x0: position of center of slit. By default 0.
            
        Post:
            Applies rectangular aperture mask to input wave.
            Updates wave.
    '''
    def rectAperture(self, D, x0=0.):
        self.U = np.where(np.abs(self.x-x0) <= D/2, self.U, 0)
      
        
    '''
        Sinusoidal amplitude grating
        (limited by a rectangular aperture)
        t = [1/2 + m/2*cos(2*pi*f0*x)]*rect(x/D)
        
        Parameters:
            - m: grating amplitude factor
            - f0: grating frequency
            - D: aperture/grating width
            
        Post: 
            Applies sinusoidal grating to input plane wave
            Updates wave
    '''
    def sinAmplitudeGrating(self, m, f0, D):
        self.U = np.where(np.abs(self.x) <= D/2, self.U, 0) # rect aperture
        self.U *= 1/2 + m/2*np.cos(2*np.pi*f0*self.x); # amplitude grating
        

    '''
        Double slit
        t = rect((x-a/2)/D) + rect((x+a/2)/D)
        
        Parameters: 
            - a: distance between (center of) slits
            - D: slit width
            
        Post:
            Applies double slit to input wave
            Updates wave
    '''
    def doubleSlit(self, a, D):
        slit1 = np.where(np.abs(self.x-a/2) <= D/2, self.U, 0)
        slit2 = np.where(np.abs(self.x+a/2) <= D/2, self.U, 0)
        self.U = slit1 + slit2


    '''
        Rectangular amplitude grating
        
        Parameters:
            - a: grating period
            - ff: fill factor. By default 0.5
            - x0: position shift. By default 0.
            
        Post:
            Applies rectangular amplitude grating to input wave
            Updates wave
            
        Note:
            Propagation gives weird results if an aperture is not applied to the field.
            (checked with uniform plane wave)
    '''
    def rectAmplitudeGrating(self, a, ff=0.5, x0=0.):
        x = self.x - x0
        t = np.cos(2*np.pi/a*x)
        f = np.cos(np.pi * ff)
        
        G = np.ones(len(self.x))
        G[t < f] = 0
        
        self.U *= G

        
    '''
        Rectangular phase grating
        t = exp(-1j*phi*G)
        G = rectangular amplitude grating
        
        Parameters:
            - P: grating period
            - phi: phase shift
            - ff: fill factor / duty cycle. By default 0.5
            - x0: position shift. By default 0.
            
        Post:
            Applies rectangular phase grating to input wave
            Updates wave
    '''
    def rectPhaseGrating(self, P, phi, ff=0.5, x0=0.):
        x = self.x - x0
    
        # Binary amplitude grating
        t = np.cos(2*np.pi/P*x)
        f = np.cos(np.pi * ff)
        G = np.ones(len(self.x))
        G[t < f] = 0
    
        # Apply amp. gr. to phase
        self.U *= np.exp(-1j*phi*G)
       
        
    '''
        Transparent lens
        
        Thin lens + paraxial approximation.
        
        Parameters:
            - f (double): focal length of lens
            - x0 (double): center of lens. By default 0 (origin).
         
         Post:
            Applies effect of lens to input wave.
            Updates field
    '''
    def lens(self, f, x0=0):
        k = 2*np.pi/self.wvl
        h = (self.x-x0)**2/(2*f)
        self.U *= np.exp(-1j*k*h)
        
        
    '''
        Trapezoidal phase function

        Test object for phase imaging.
        
        Parameters:
            - phi = max phase change
            - a = position of constant to linear phase change
            - b = position of linear to null phase change
     '''
    def trapezoidPhaseObject(self, phi, a, b):
        R = (b-np.abs(self.x))/(b-a) # pyramid function, R(a)=1, R(b)=0
        G = np.where(np.abs(self.x) <= b, R, 0)
        G[np.abs(self.x) <= a] = 1 # truncate pyramid to make trapezoid
        
        self.U *= np.exp(-1j*phi*G)

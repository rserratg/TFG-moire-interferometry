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
            
        Post:
            Applies rectangular aperture mask to input plane wave.
            Updates wave.
    '''
    def rectAperture(self, D):
        self.U = np.where(np.abs(self.x) <= D/2, self.U, 0)
      
        
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
            Applies double slit to input plane wave
            Updates wave
    '''
    def doubleSlit(self, a, D):
        slit1 = np.where(np.abs(self.x-a/2) <= D/2, self.U, 0)
        slit2 = np.where(np.abs(self.x+a/2) <= D/2, self.U, 0)
        self.U = slit1 + slit2


    '''
        Rectangular amplitude grating
        Every period: half 1, half 0
        
        Parameters:
            - a: grating period
            
        Post:
            Applies rectangular phase grating to input plane wave
            Updates wave
            
        Note:
            Propagation gives weird results if an aperture is not applied to the field.
            (checked with uniform plane wave)
    '''
    def rectAmplitudeGrating(self, a):
        G = np.cos(2*np.pi/a*self.x) >= 0
        self.U *= G

        
    '''
        Rectangular phase grating
        t = exp(-1j*phi*G)
        G = rectangular amplitude grating (for every period: half 1, half 0)
        
        Parameters:
            - phi: phase shift
            - P: grating period
            
        Post:
            Applies rectangular phase grating to input plane wave
            Updates wave
    '''
    def rectPhaseGrating(self, phi, P):
        G = np.cos(2*np.pi/P*self.x) >= 0
        self.U *= np.exp(-1j*phi*G)
    

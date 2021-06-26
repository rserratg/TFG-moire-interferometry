# Optical elements for the NeutronWave class

import numpy as np

'''
    Helper (Mixin) class for the NeutronWave class.
    
    Contains the method related to optical elements.
    
    Note:
        - Elements should use global space self.X.
'''

class MixinElem:

    '''
        Slit (rectangular aperture)
        t = rect(x/D)
        
        Parameters:
            - D (double): aperture width
            - x0 (double): position of center of slit. By default 0.
    '''
    def slit(self, D, x0=0.):
        self.Psi = np.where(np.abs(self.X - x0) <= D/2, self.Psi, 0)
        
    
    '''
        Double slit
        t = rect((x-a/2)/D) + rect((x+a/2)/D)
        
        Parameters:
            - a (double): distance between (center of) slits
            - D (double): slit width
    '''
    def doubleSlit(self, a, D):
        slit1 = np.where(np.abs(self.X-a/2) <= D/2, self.Psi, 0)
        slit2 = np.where(np.abs(self.X+a/2) <= D/2, self.Psi, 0)
        self.Psi = slit1 + slit2
        
    
    '''
        Binary phase grating
        t = exp(-1j*phi*G)
        G = binary amplitude grating
        
        Parameters:
            - P (double): grating period
            - phi (double): phase shift
            - ff (double): fill factor / duty cycle. By default 0.5.
            - x0: position shift. By default 0.
    '''
    def rectPhaseGrating(self, P, phi, ff=0.5, x0=0.):
        X = self.X - x0
        
        # Binary amplitude grating
        t = np.cos(2*np.pi/P*X)
        f = np.cos(np.pi * ff)
        G = np.ones_like(self.Psi)
        G[t < f] = 0
        
        # Apply amp. gr. to phase
        self.Psi *= np.exp(-1j*phi*G)

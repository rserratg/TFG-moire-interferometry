# Optical elements for the OptWave2 class

import numpy as np

'''
    Helper (Mixin) class for the OptWave2D class

    Contains the methods related to optical elements
'''

class MixinElem2D:
    
    '''
        Rectangular slit
        t = rect(x/Lx)*rect(y/Ly)

        Parameters:
            - Lx (double): slit width in x
            - Ly (double): slit width in y

        Post:
            Applies rectangular mask to input wave.
    '''
    def rectSlit(self, Lx, Ly):
        cond1 = np.abs(self.x) <= Lx/2
        cond2 = np.abs(self.y) <= Lx/2
        C1, C2 = np.meshgrid(cond1, cond2)
        self.U = np.where(C1 & C2, self.U, 0)


    '''
        Binary amplitude grating: ||||

        Periodic in x direction.
        Constant in y direction.

        Parameters:
            - a (double): grating period
            - ff (double): fill factor. By defalt 0.5.
            - x0 (double): position shift in x direction. By default 0.

        Post:
            Applies grating mask to input wave.
    '''
    def rectAmplitudeGratingX(self, a, ff=0.5, x0=0.):
        x = self.x - x0

        t = np.cos(2*np.pi/a*x)
        f = np.cos(np.pi * ff)

        self.U[:,t<f] = 0


    '''
        Binary phase grating

        Periodic in x direction.
        Constant in y direction.

        tp = exp(-1j*phi*G)
        G = binary amplitude grating in x direction

        Parameters: 
            - P (double): grating period
            - phi (double): phase shift
            - ff (double): fill factor / duty cycle. By default 0.5.
            - x0 (double): position shift in x direction. By default 0.

        Post:
            Applies grating mask to input wave.
    '''
    def rectPhaseGratingX(self, P, phi, ff=0.5, x0=0.):
        x = self.x - x0

        # Binary amplitude grating (1d)
        t = np.cos(2*np.pi/P*x)
        f = np.cos(np.pi * ff)
        G = np.ones(len(self.x))
        G[t<f] = 0

        # Binary phase grating (1d)
        tp = np.exp(-1j*phi*G)

        # Apply to each row
        self.U *= tp

    '''
        Transparent lens

        Thin lens + paraxial approximation

        Parameters:
            - f (double): focal length of lens
            - x0 (double): center of lens in x. By default 0.
            - y0 (double): center of lens in y. By default 0.

        Post:
            Applies lens mask to input wave.
    '''
    def lens(self, f, x0=0., y0=0.):
        x = self.x - x0
        y = self.y - y0
        k = 2*np.pi/self.wvl 

        x2 = x**2
        y2 = y**2

        X2, Y2 = np.meshgrid(x2, y2)
        R2 = X2 + Y2

        H = R2/(2*f)
        self.U *= np.exp(-1j*k*H)

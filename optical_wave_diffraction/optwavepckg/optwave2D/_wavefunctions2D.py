# Wavefunction initializers for OptWave2D class

import numpy as np

'''
    Helper (Mixin) class for the OptWave2D class

    Contains methods related to generation of wavefunctions
'''

class MixinWave2D:

    '''
        Uniform plane wave

        Propagation parallel to z axis.
        self.U = A

        Parameters:
            - A (double): amplitude of the wave. By default 1.

        Post:
            - self.U is a plane wave with amplitude A.
    '''
    def planeWave(self, A=1.):
        self.U = A * np.ones_like(self.U)


    '''
        Gaussian beam

        Propagation parallel to z axis.

        Parameters:
            - w0 (double): minimum beam width
            - x0 (double): position of center in x. By default 0.
            - y0 (double): position of center in y. By default 0.
            - z0 (double): position w.r.t. beam waist. By default 0.
                           Positive for divergent beam (right of beam waist)
                           Negative for convergent beam (left of beam waist)
            - A (double): maximum amplitude. By default 1.
        
        Post:
            - self.U is a gaussian beam with the given parameters
    '''
    def gaussianBeam(self, w0, x0=0., y0=0., z0=0., A=1.):
        k = 2 * np.pi / self.wvl

        # Rayleigh range
        zr = k * w0**2 / 2

        # Gouy phase
        gouy = np.arctan2(z0, zr)

        w = w0 * np.sqrt(1+(z0/zr)**2)
        if z0 == 0:
            R = 1e10 # At beam waist, R = infinity
        else:
            R = z0 * (1 + (zr/z0)**2)

        x2 = (self.x-x0)**2
        y2 = (self.y-y0)**2
        X2, Y2 = np.meshgrid(x2, y2)
        rho2 = X2 + Y2

        amplitude = A * (w0/w) * np.exp(-rho2/w**2)
        phase1 = np.exp(1j * (k*z0 - gouy))
        phase2 = np.exp(1j * k * rho2 / (2*R))

        self.U = amplitude * phase1 * phase2

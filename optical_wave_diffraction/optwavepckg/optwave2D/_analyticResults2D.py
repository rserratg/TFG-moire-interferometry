import numpy as np
import scipy.special as sp 

'''
    Helper (Mixin) class for the OptWave2D class

    Contains methods that provide analytical solutions
'''

class MixinAR2D:

    '''
        Plane wave of unit amplitude
        Rectangular aperture
        Fresnel propagation
        (2D)

        Parameters:
            - z (double): propagation distance from aperture
            - Lx: aperture width in x
            - Ly: aperture width in y
    '''
    def planeRectFresnelSolution(self, z, Lx, Ly):
        A = np.sqrt(self.wvl*z/2)
        k = 2*np.pi/self.wvl

        alpha1 = (self.x + Lx/2)/A
        alpha2 = (self.x - Lx/2)/A
        beta1 = (self.y + Ly/2)/A
        beta2 = (self.y - Ly/2)/A

        sa1, ca1 = sp.fresnel(alpha1)
        sa2, ca2 = sp.fresnel(alpha2)
        sb1, cb1 = sp.fresnel(beta1)
        sb2, cb2 = sp.fresnel(beta2)

        B = A**2 * np.exp(1j*k*z)/(1j*self.wvl*z)
        Ix = (ca1-ca2) + 1j*(sa1-sa2)
        Iy = (cb1-cb2) + 1j*(sb1-sb2)

        Ixx, Iyy = np.meshgrid(Ix, Iy)
        Uout = B*Ixx*Iyy
        return Uout

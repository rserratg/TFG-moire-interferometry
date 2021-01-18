import numpy as np

from . import _analyticResults, _elements, _propagation, _wavefunctions

'''
    Class OptWave represents a 1D optical wave
    
    Properties:
        - d: x-space sampling interval
        - x: x-space sampling points
        - wvl: wavelength
        - U: wave function (complex) amplitude at sampling points
'''
class OptWave(
    _analyticResults.MixinAR,
    _elements.MixinElem,
    _propagation.MixinProp,
    _wavefunctions.MixinWave
):

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
        #self.x = np.linspace(-L/2,L/2,N)
        #self.d = self.x[1]-self.x[0]
        self.wvl = lamb
        self.U = np.zeros(N, dtype=np.complex128)

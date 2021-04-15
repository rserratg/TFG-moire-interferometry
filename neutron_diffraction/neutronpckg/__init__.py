import numpy as np
from . import _elements, _propagation, _wavefunctions

'''
    Class NeutronWave represents a neutron matter-wave
    
    Properties:
        - Sampling space:
            - Nx (int): number of sampling points
            - d (double): x-space sampling interval
            - x (np.array, Nx): x-space sampling points (local space)
        - Neutron properties:
            - Nn (int): number of neutrons
            - wvl (double): wavelength
            - theta (np.array, Nn): angle of propagation of each neutron w.r.t. z axis
            - x0 (np.array, Nn): center of each neutron's space
        - Experiment:
            - L (double): total propagation distance (from source)
            - Psi (np.array, Nn x Nx): wave function (complex) amplitude at sampling points
            - X (np.array, Nn x Nx): local sampling space of each neutron in global space
                                     coordinates (X = x + x0 + L*tan(theta))
'''
class NeutronWave(
    _elements.MixinElem,
    _propagation.MixinProp,
    _wavefunctions.MixinWave
):

    '''
        Default constructor.
        
        Parameters:
            - Nx: number of grid points
            - Sx: size of x-space
            - wvl: wavelength. By default 4.4e-10
            - Nn: number of neutrons. By default 1000.
    '''
    def __init__(self, Nx, Sx, wvl=4.4e-10, Nn=1000):
    
        # Cast to int, allows to use float exponential notation
        Nx = int(Nx)
        
        self.Nx = Nx
        self.d = Sx/Nx
        self.x = np.arange(-Nx//2, Nx//2)*self.d
        
        self.Nn = Nn
        self.wvl = wvl
        self.theta = np.zeros((Nn,1))   # column vector
        self.x0 = np.zeros((Nn,1))      # column vector
        
        self.L = 0
        self.Psi = np.zeros((Nn, Nx), dtype=np.complex128)
        self.X = np.tile(self.x, (Nn,1))

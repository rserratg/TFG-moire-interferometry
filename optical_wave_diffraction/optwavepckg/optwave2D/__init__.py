import numpy as np

from . import _analyticResults2D, _elements2D, _propagation2D, _wavefunctions2D

'''
    Class OptWave2D represents a 2D optical wave

    Properties:
        - dx: x-space sampling interval
        - dy: y-space sampling interval
        - x: x-space sampling points
        - y: y-space sampling points
        - wvl: wavelength
        - U: wave function (complex) amplitude at sampling points (in xy-space)
'''

class OptWave2D(
    _analyticResults2D.MixinAR2D,
    _elements2D.MixinElem2D,
    _propagation2D.MixinProp2D,
    _wavefunctions2D.MixinWave2D
):

    '''
        Default constructor

        Parameters:
            - N (2-int tuple or int): 
                Number of grid points in x and y
                If int, Nx = Ny
            - L (2-double tuple or double):
                Size of space in x and y
                If int, Lx = Ly
            - wvl (double): wavelength
            - symmetric: 
                If True, use symmetric x- and y-space.
                By default, False.

        Post:
            Initialize class properties according to input parameters.
            U initialized to zero.

        Notes:
            - U coordinates are consistent with meshgrid(x,y). 
              [
                  [(x0,y0), (x1,y0), (x2,y0)...],
                  [(x0,y1), (x1,y1), (x2,y1)...],
                  ...
              ]
            - meshgrid(X,Y) is calculated when needed.
              Precalculating is not worth it.
              Operation like squaring make it better to do the meshgrid as late as possible.

    '''
    def __init__(self, N, L, wvl, symmetric=False):
        
        # Read tuple or int N
        try:
            Nx, Ny = N
            Nx = int(Nx)
            Ny = int(Ny)
        except TypeError:
            Nx = Ny = int(N)

        # Read tuple or int L
        try:
            Lx, Ly = L
        except:
            Lx = Ly = L

        # Init x and y space
        if symmetric:
            self.x = np.linspace(-Lx/2, Lx/2, Nx)
            self.dx = self.x[1] - self.x[0]
            self.y = np.linspace(-Ly/2, Ly/2, Ny)
            self.dy = self.y[1] - self.y[0]
        else:
            self.dx = Lx/Nx
            self.x = np.arange(-Nx//2, Nx//2)*self.dx
            self.dy = Ly/Ny
            self.y = np.arange(-Ny//2, Ny//2)*self.dy
        
        # Init wavelength
        self.wvl = wvl

        # Init field
        self.U = np.zeros((Ny,Nx), dtype=np.complex128)


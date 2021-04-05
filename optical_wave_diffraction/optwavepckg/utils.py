# Utility functions for the OptWave and OptWave2 classes

import numpy as np 
from scipy.optimize import curve_fit    
    
'''
    Field normalization to amplitude/intensity 1
    
    Parameters:
        - u (numpy array): field amplitude or intensity
        
    Post:
        Returns normalized field
        
'''
def normalize(u):
    norm = (np.abs(u)).max()
    return u/norm
    
    
'''
    Field intensity
    
    Parameters:
        - u (numpy array): field amplitude
        
    Post:
        Returns field intensity
'''
def intensity(u):
    return np.abs(u)**2
    
    
'''
    Normalized field intensity
    
    Parameters:
        - u (numpy array): field amplitude
        
    Post:
        Returns normalized field intensity
        
    Note:
        - Difference w.r.t. to doing normalize(intensity(u)) is that it has smaller chance
          to produce overflow (square before vs. after normalization).
'''
def normalizedIntensity(u):
    return normalize(np.abs(u))**2
    

'''
    Rebin field to given number of pixels
    (1D)
    
    Parameters:
        - x (np.array): x-space
        - u (np.array): 1D field to bin
        - N (int): number of bins
        - fast (bool): 
            If True, use reshape method. 
            If False, use numpy histogram. 
            By default False.
        - avg (bool):
            If True, use average over each bin.
            If False, use sum over each bin.
            By default False.
        
    Returns:
        - xmod (np.array): rebinned x-space
        - umod (np.array): rebinned field
        
    Note:
        - Reshape method is significantly faster for very large arrays
        - Reshape method requires that N divides the number of points in x.
        - It is assumed that len(x) = len(y)
        - All bins except the last (right-most) are half-open.
        - Size of pixels can be obtained as: pixel_size = xmod[1]-xmod[0]
'''
def rebin(x, u, N, fast=False, avg=False):

    N = int(N)
    
    if fast:
    
        M = len(u)
        
        # Check validity of N
        if M % N != 0:
            print("Rebin - reshape: N must divide the number of points in the field")
            return  
            
        # Get new shape
        shape = (N, M//N)
        
        # Reshape array and average / sum each bin
        xmod = x.reshape(shape).mean(1)
        umod = u.reshape(shape)
        umod = umod.mean(1) if avg else umod.sum(1)
    
    else:
    
        umod, xedges = np.histogram(x, N, weights=u) # umod = sum
        if avg:
            umod /= np.histogram(x, N)[0] # Normalize -> umod = average
        xmod = (xedges[1:]+xedges[:-1])/2
    
    return xmod, umod


'''
    Rebin field to given number of pixels
    (2D)
    
    Parameters:
        - x (np.array): x-space
        - y (np.array): y-space
        - u (np.array): 2D field to bin
        - bins (int or 2-int tuple):
            If int, the number of bins for the two dimensions (Nx = Ny = bins)
            If tuple, the number of bins for each dimension (Nx, Ny = bins)
        - fast (bool):
            If True, use reshape method.
            If False, use numpy histogram.
            By default False.
        - avg (bool):
            If True, average over each bin
            If False, sum over each bin
            By default False.
            
            
    Returns:
        - xmod (np.array): rebinned x-space
        - ymod (np.array): rebinned y-space
        - umod (np.array): rebinned field
        
    Notes:
        - Reshape method is significantly faster for large arrays-
        - Reshape method requires that Nx (Ny) divides the number of points in x (y)
        - It is assumed that shape(u) = (len(y), len(x))
'''            
def rebin2(x, y, u, bins, fast=False, avg=False):

    # Check if N is int or tuple
    # Cast to int for safety
    try:
        Nx, Ny = bins
        Nx = int(Nx)
        Ny = int(Ny)
    except TypeError:
        Nx = Ny = int(bins)
        
    if fast:
        
        Mx = len(x)
        My = len(y)
        
        # Check validity of bins
        if (Mx % Nx != 0) or (My % Ny != 0):
            print('Rebin - reshape: Nx (Ny) must divide the number of points in x (y)')
            return
        
        # Get new shape
        shape_x = (Nx, Mx//Nx)
        shape_y = (Ny, My//Ny)
        shape = (Ny, My//Ny, Nx, Mx//Nx)
        
        # Reshape arrays and get average of each bin
        xmod = x.reshape(shape_x).mean(1)
        ymod = y.reshape(shape_y).mean(1)
        umod = u.reshape(shape)
        umod = umod.mean(-1).mean(1) if avg else umod.sum(-1).sum(1)
        
    else:
        
        # Prepare data for histogram
        xaux = np.tile(x, len(y))
        yaux = np.repeat(y, len(x))
        uaux = u.flatten()
        
        umod, xedges, yedges = np.histogram2d(xaux, yaux, (Nx,Ny), weights=uaux) # umod = sum
        if avg:
            umod /= np.histogram2d(xaux, yaux, (Nx,Ny))[0] # umod = mean
        umod = umod.transpose()
        xmod = (xedges[1:]+xedges[:-1])/2
        ymod = (yedges[1:]+yedges[:-1])/2
        
    return xmod, ymod, umod
    

'''
    Contrast from periodic (sinusoidal) fringes with known period
    (1D)
    
    Assuming U(x) = A + B*sin(2*pi*x/P + phi)
    
    Parameters:
        - x (np.array): x-space
        - u (np.array): intensity of field at observation plane
        - P (double): expected fringe period
        - xlim (2-double tuple): interval to which the curve is fitted. 
            By default all the x-space is used.
        - retfun: if True return fitted function evaluated at x. By default False.
        
    Returns:
        - C (double): Contrast of periodic fringes
        - fit (np.array): fitted function evaluated at x. Only if retfun = True.
        
    Notes:
        - To calculate contrast of a modulated field:
            - Take intensity of field at the observation plane (I)
            - Take intensity of reference field at the observation plane (Iref).
              That is, the result without the object (e.g. grating) under study.
            - Obtain the fringe pattern by dividing them: u = I/Iref
'''
def contrast(x, u, P, xlim = None, retfit = False):

    
    # Get only data inside interval limited by xlim
    xaux = x
    if xlim is not None:
        xmin, xmax = xlim
        cond1 = x >= xmin
        cond2 = x <= xmax
        cond = cond1 & cond2
        xaux = x[cond]
        u = u[cond]
    
    # Sin function to fit
    def fun(xx, a, b, c):
        return a + b*np.sin(2*np.pi*xx/P + c)
    
    # Fit function to data and retrieve optimal parameters
    popt, _ = curve_fit(fun, xaux, u)
    A, B, phi = popt
    
    # Calculate contrast
    C = abs(B/A)
    
    if retfit:
        fit = fun(x, A, B, phi)
        return C, fit
    else:
        return C
        
# WARNING: TEST FUNCTIONS
'''
    Contrast and period from periodic (sinusoidal) fringes
    (1D)
    
    Assuming U(x) = A + B*sin(2*pi*x/P + phi)
    
    Parameters:
        - x (np.array): x-space
        - u (np.array): intensity of field at observation plane
        - P0 (double): initial guess for period
        - xlim (2-double tuple): interval to which the curve is fitted. 
            By default all the x-space is used.
        - retfun: if True return fitted function evaluated at x. By default False.
        
    Returns:
        - C (double): Contrast of periodic fringes
        - P (double): period of fringes
        - sdP (double): standard deviation of period value
        - fit (np.array): fitted function evaluated at x. Only if retfun = True.
        
    Notes:
        - Initial guess must be sufficiently close.
          Otherwise fit will probably fail to converge to the proper value.
'''
def contrast_period(x, u, P0, xlim = None, retfit = False):

    
    # Get only data inside interval limited by xlim
    xaux = x
    if xlim is not None:
        xmin, xmax = xlim
        cond1 = x >= xmin
        cond2 = x <= xmax
        cond = cond1 & cond2
        xaux = x[cond]
        u = u[cond]
    
    # Sin function to fit
    def fun(xx, a, b, c, d):
        return a + b*np.sin(2*np.pi*xx/d + c)
    
    # Fit function to data and retrieve optimal parameters
    p0 = (np.mean(u), np.std(u), 0, P0)
    bmin = [0, 0, -np.pi, 0]
    bmax = [np.inf, np.inf, np.pi, np.inf]
    popt, pcov = curve_fit(fun, xaux, u, p0=p0, bounds=(bmin, bmax))
    A, B, phi, P = popt
    
    sdP = np.sqrt(np.diag(pcov))[3]
    
    # Calculate contrast
    C = abs(B/A)
    
    if retfit:
        fit = fun(x, A, B, phi, P)
        return C, P, sdP, fit
    else:
        return C, P, sdP
        

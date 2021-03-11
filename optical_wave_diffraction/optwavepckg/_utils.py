# Utility functions for the OptWave class

# Note: this functions are not a part of the class

import numpy as np

'''
    1D Fourier Transform
    
    Parameters:
        - g: x-space amplitudes
        - delta: sampling interval
        
    Post:
        Returns freq-space amplitudes (DFT of g)
        
    Note:
        - Shifted in x-space and freq-space
'''
def ft(g, delta=1):
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g)))
    return G*delta
    
    
'''
    1D Inverse Fourier Transform
    
    Parameters:
        - G: x-space amplitudes
        - delta_f: frequency spacing
            if delta_f = None, 1/N is used (default numpy normalization)
        
    Post:
        Returns x-space amplitudes (IDFT of G)
        
    Notes:
        - Numpy normalizes the ifft by 1/N. 
        - Shifted in freq-space and x-space
'''
def ift(G, delta_f=None):
    g = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G)))
    if delta_f is None:
        return g
    else:
        return g*len(g)*delta_f
    
    
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
    Visibility of fringes from a grating
    
    V = (Imax-Imin)/(Imax+Imin)
    Imax = integration between -P/4 and P/4
    Imin = integration between P/4 and 3P/4
    
    This definition avoids experimental fluctuations
    
    Parameters:
        - I (numpy array): intensity pattern
        - x (numpy array): x-space
        - P (float): grating period
'''
def visibility(I, x, P):
    # 1 period
    I1 = I[(x >= -P/4) & (x<=P/4)]
    I2 = I[(x >= P/4) & (x<=3*P/4)]
    
    Imax = I1.sum()
    Imin = I2.sum()
    
    V = (Imax-Imin)/(Imax+Imin)
    return abs(V)
    
'''
    Binned field
    Each point takes the value of the integrated field over a certain interval.
    If N % len(I) != 0, last interval is shorter
    
    Parameters:
        - I (numpy array): intensity pattern
        - x (numpy array): x-space
        - N (int): interval size in number of points
        
    Returns:
        - binned intensity pattern
        - new x-space (left edges of bins)
        
    Notes:
        - Plot with
            newI, newX = binning(I,x,N)
            plt.bar(newX, newI, width=newX[1]-newX[0],align='edge')
'''
def binning(I, x, N):
    newI = np.add.reduceat(I, range(0, len(I), N))
    newX = x[::N]
    return newI, newX

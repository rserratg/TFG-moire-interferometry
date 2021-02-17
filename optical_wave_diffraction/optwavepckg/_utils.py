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

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
        - Shift on freq-space must be done to apply frequencies -fx:fx
        - Shift on x-space makes results smoother in phase
'''
def ft(g, delta=1):
    G = np.fft.fftshift(np.fft.fft(np.fft.fftshift(g)))
    return G*delta
    
    
'''
    1D Inverse Fourier Transform
    
    Parameters:
        - G: x-space amplitudes
        - delta_f: frequency spacing
        
    Post:
        Returns x-space amplitudes (IDFT of G)
        
    Notes:
        Numpy normalizes the ifft by 1/N. 
'''
def ift(G, delta_f=1):
    g = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G)))
    return g*len(g)*delta_f

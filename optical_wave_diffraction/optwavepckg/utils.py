# Utility functions for the OptWave and OptWave2 classes

import numpy as np 
    
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
    
    
# TEST FUNCTIONS
    
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

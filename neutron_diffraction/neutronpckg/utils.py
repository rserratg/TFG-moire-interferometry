# Utility functions for simulation of neutron diffraction

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


'''
    Calculate contrast by fitting sine function to data.
    
    Parameters:
        - x (np.array): x-space
        - u (np.array): data to fit
        - P0 (double): expected period
        - xlim (two-double tuple): region of x-space to use. By default all x-space is used.
        - fitP (bool): 
            If True, period is a free-parameter in fit. P0 is the initial guess.
            If False, the period of the sine is fixed to P0.
            By default False.
            
    Returns:
        - C (double): contrast
        - P (double): period. If not fitP, P0 is returned.
        - fit (np.array): fitted data
'''
def contrast_fit(x, u, P0, xlim = None, fitP = False):

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
    # TODO: initial guesses
    if fitP:
        p0 = (np.mean(u), np.std(u), 0, P0)
        bmin = [0,0,0,0]
        bmax = [np.inf,np.inf,2*np.pi,np.inf]
        popt, _ = curve_fit(fun, xaux, u, p0=p0, bounds=(bmin,bmax))
        A, B, phi, P = popt
    else:
        p0 = (np.mean(u), np.std(u), 0)
        popt, _ = curve_fit(lambda xx,a,b,c : fun(xx,a,b,c,P0), xaux, u, p0=p0)
        A, B, phi = popt
        P = P0
        
    # Calculate contrast and fitted data over the whole space
    C = abs(B/A)
    fit = fun(x, A, B, phi, P)
    return C, P, fit
    

'''
    Calculate contrast using FFT
    
    Finds frequency f of the peak closer to f0.
    If f0 is not provided, f is the frequency with largest amplitude.
    Contrast is calculated as 2*abs(FT[f])/abs(FT[0])
    
    Params:
        - d (double): sampling space period
        - u (np.array): data to calculate contrast
        - f0 (double): expected frequency of fringes. By default the higher peak is considered.
        - fmax (double): maximum frequency to consider. By default None.
        - plotft (bool): If True, plot the FT up to fmax if give. By default False.
        
    Returns:
        - C (double): contrast of fringes
        - fd (double): frequency of fringes
        
    Note:
        - It is assumed that u is one-dimensional and real-valued
'''
def contrast_FT(d, u, f0=None, fmax=None, plotft=False):
    
    # Calculate FT and frequencies
    ft = np.abs(np.fft.rfft(u))
    freq = np.fft.rfftfreq(len(u), d)
    
    # Discard components aboce max freq
    if fmax is not None:
        cond = freq < fmax
        ft = ft[cond]
        freq = freq[cond]
        
    # Find indices of peaks
    # Maxima at f=0 and f=fmax are not considered peaks
    pks, _ = find_peaks(ft)
    
    # Find index of higher peak / peak closer to f0 in pks
    if f0 is None:
        idx = freq[pks].argmax()
    else:
        idx = (np.abs(freq[pks]-f0)).argmin()
    
    # Retrieve index of peak in original arrays (ft, freq)
    idx = pks[idx]
    
    # Results
    C = 2*ft[idx]/ft[0]
    fd = freq[idx]
    
    # Plot ft, position of peaks (x) and peak of interest (red dot)
    if plotft:
        plt.plot(freq, ft, '-o')
        plt.plot(freq[pks], ft[pks], 'x')
        plt.plot(freq[idx], ft[idx], 'ro')
        plt.xlabel(r'Frequency [$m^{-1}$]')
        plt.ylabel('Amplitude [arbitrary units]')
        plt.show()
        
    return C, fd

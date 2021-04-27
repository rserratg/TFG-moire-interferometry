# Plotting utils for the NeutronWave class

import numpy as np

'''
    Helper (Mixin) class for the NeutronWave class
    
    Contains the methods related to acquisition of results and plotting.
'''

class MixinDraw:

    
    '''
        Histogram intensity.
        
        Intensity is sum of all squared neutron wavefunctions.
        Results are to be plotted e.g. with pyplot.bar()
        
        Parameters:
            - numbins (int): number of bins to use in histogram
            - xlimits (2-double tuple):
                Region of x-space to consider.
                (xmin, xmax)
                If None, uses minimum and maximum values among all neutron spaces.
                By default None.
            - retcenter (bool):
                If True, returns center of bins.
                If False, returns edges of bins.
                By default False.
        
        Returns:
            - bins (np.array): 
                If retcenter, center of bins (len = numbins).
                Otherwise, edges of bins (len = numbins + 1).
            - hist (np.array): histogram data (intensity at camera, not normalized)
    '''
    def hist_intensity(self, numbins, xlimits = None, retcenter = False):
    
        # Calculate xlimits
        try:
            xmin, xmax = xlimits
        except TypeError:
            xmin = self.X.min()
            xmax = self.X.max()
            
        x = self.X
        I = np.abs(self.Psi)**2
        
        bins = np.linspace(xmin, xmax, numbins)
        hist = np.zeros(numbins-1, dtype=np.double)
        for xx, ii in zip(x, I):
            htemp, _ = np.histogram(xx, bins, weights=ii)
            hist += htemp
            
        if retcenter:
            bins = (bins[:-1] + bins[1:])/2
            
        return bins, hist
        

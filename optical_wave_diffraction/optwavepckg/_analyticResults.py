# Analytic examples for the OptWave class

# Note: this functions return the wavefunction for comparison purposes.
#       The wavelength and sampling points are taken from the object.
#       They do NOT modify the object's wavefunction.

import numpy as np
import scipy.special as sp

'''
    Helper (Mixin) class for the OptWave class.
    
    Contains methods that provide analytical solutions.
'''

class MixinAR:

    '''
        Plane wave of unit amplitude
        Rectangular aperture
        Fraunhofer propagation
        (1D)
        
        Parameters:
            - z: propagation distance from aperture
            - Lx: aperture width
            
        Pre:
            Lx <= 2*x_max
            
        Post:
            Returns analytic solution sampled at x
            
        Notes:
            np.sinc(x) = sin(pi*x)/(pi*x)
    '''
    def planeRectApertureSolution(self, z, Lx):
        k = 2*np.pi/self.wvl
        A = np.exp(1j*k*z)/np.sqrt(1j*self.wvl*z)
        B = np.exp(1j*self.x**2/(2*z))
        Uout = A*B*Lx*np.sinc(Lx*self.x/self.wvl/z)
        return Uout
        
        
    '''
        Plane wave of unit amplitude
        Rectangular aperture
        Fresnel propagation
        (1D)
        
        Parameters:
            - z: propagation distance from aperture
            - L: aperture width
    '''
    def planeRectFresnelSolution(self, z, L):
        A = np.sqrt(self.wvl*z/2)
        k = 2*np.pi/self.wvl

        #substitutions
        alpha1 = (self.x + L/2)/A
        alpha2 = (self.x - L/2)/A
        
        # Fresnel sine and cosine integrals
        sa1, ca1 = sp.fresnel(alpha1)
        sa2, ca2 = sp.fresnel(alpha2)
        
        # Observation-plane field
        B = -A*np.exp(1j*k*z)/np.sqrt(1j*self.wvl*z)
        Uout = B*((ca2-ca1) + 1j*(sa2-sa1))
        return Uout
        
    '''
        Plane wave of unit amplitude
        Sinusoidal amplitude grating limited by rectangular aperture
        Fraunhofer propagation 
        (1D)
        
        Parameters:
            - z: propagation distance from grating 
            - m: grating amplitude factor
            - f0: grating frequency
            - D: aperture width
            
        Pre:
            D <= 2*x_max
            
        Post:
            Returns analytic solution sampled at x
    '''
    def planeSinAmpGrAnalyticSolution(self, z, m, f0, D):
        k = 2*np.pi/self.wvl;
        lz = self.wvl*z;
        Uout = np.sinc(D*self.x/lz)  # central sinc
        Uout += m/2*np.sinc(D*(self.x-f0*lz)/lz) # right sinc
        Uout += m/2*np.sinc(D*(self.x+f0*lz)/lz) # left sinc
        Uout = Uout*D/2*np.exp(1j*k*z*(1+self.x**2/(2*z**2)))/(1j*lz)
        return Uout
       
        
    '''
        Plane wave of unit amplitude
        Double slit
        Fraunhofer propagation
        (1D)
        
        Parameters:
            - z: propagation distance from slits plane
            - a: distance between slits (center to center)
            - D: slit width
            
        Pre: 
            Slits should be within x-space
            
        Post:
            Returns analytic solution sampled at x
    '''
    def planeDoubleSlitAnalyticSolution(self, z, a, D):
        k = 2*np.pi/self.wvl
        lz = z*self.wvl
        Uout = 2*D*np.sinc(D*self.x/lz)*np.cos(np.pi*a*self.x/lz)
        Uout = Uout*np.exp(1j*k*(z+self.x**2/(2*z)))/(1j*lz)
        return Uout
        
        
    '''
        Plane wave of unit amplitude
        Rectangular amplitude grating limited by rectangular aperture
        Fraunhofer propagation 
        (1D)
        
        Parameters:
            - z: propagation distance from grating 
            - a: grating period
            - L: aperture width
            
        Pre:
            L <= 2*x_max
            
        Post:
            Returns analytic solution sampled at x
            
        Note:
            The rectangular amplitude grating element is not limited by any aperture,
            however the field itself is limited to 2*x_max.
    '''
    def planeRectAmpGrAnalyticSolution(self, z, a, L):
        k = 2*np.pi/self.wvl;
        lz = self.wvl*z;
        
        Uout = 0
        for n in range(-10,11):
            Uout += np.sinc(L*(self.x-n*lz/a)/lz)
        Uout *= L/2*np.sinc(self.x*a/(2*lz))
        
        Uout = Uout*np.exp(1j*k*z*(1+self.x**2/(2*z**2)))/(1j*lz)
        return Uout

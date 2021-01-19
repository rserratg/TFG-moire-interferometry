# Propagation methods for the OptWave class

import numpy as np
from ._utils import ft, ift
import matplotlib.pyplot as plt
from scipy.special import hankel1

'''
    Helper (Mixin) class for the OptWave class.
    
    Contains the methods related to wave propagation methods.
'''


class MixinProp:

    # AUXILIARY METHODS: CONVOLUTION CALCULATIONS

    '''
        Propagate field using impulse response
        (Convolution fft calculation for propagation methods)
        Using zero-padding and, optionally, Simpson's rule.

        S = IFFT[FFT(U)*FFT(H)]

        Parameters:
            - kernel (function(x)): impulse response, parameter x is x-space
            - simpson (bool): 
                if True, uses Simpson's rule for better acuracy
                True by default

        References:
            - Ji Qiang. A high-order fast method for computing convolution integral with smooth kernel.
    '''
    def _fft_conv_(self, kernel, simpson=True):
        
        N = len(self.x)
        x = self.x
        u = self.U

        # Simpson's integration rule
        if simpson:
            a = [2,4]
            num_rep = int(round(N/2)-1)
            b = np.array(a * num_rep)
            W = np.concatenate(((1,), b, (2,1))) / 3.
            if float(N) / 2 == round(N/2): # is even
                i_central = num_rep + 1
                W = np.concatenate((W[:i_central], W[i_central+1:]))
            u = W*u

        # Zero-padding
        
        # Field
        U = np.zeros(2*N-1, dtype=complex)
        U[0:N] = np.array(u)

        # Double domain
        xext = self.x[0] - x[::-1]
        xext = xext[0:-1]
        x = np.concatenate((xext, self.x - x[0]))

        # H = kernel in freq-space
        H = np.fft.fft(kernel(x))

        S = np.fft.ifft(np.fft.fft(U)*H) * self.d
        self.U = S[N-1:]

    '''
        Propagate field using transfer function.
        (Convolution fft calculation for propagation methods)
        Using zero-padding and, optionally, Simpson's rule

        S = IFFT[FFT(U)*H]

        Parameters:
            - kernel (function(fx)): transfer function, parameter fx is freq-space
            - bandlimit (False or double):
                if double, delete frequency components above given limit
                if False, don't delete any frequency component
                False by default
            - simpson (bool): 
                if True, uses Simpson's rule for better accuracy
                True by default

        Notes:
            - If bandlimit=0 it is treated as if it were False.
            - Important: notice that the domain is double due to zero-padding.
              Therefore, df'=2*df
    '''
    def _ifft_conv_(self, kernel, bandlimit=False, simpson=True):
        
        N = len(self.x)
        u = self.U

        # Simpson's integration rule
        if simpson:
            a = [2,4]
            num_rep = int(round(N/2)-1)
            b = np.array(a * num_rep)
            W = np.concatenate(((1,), b, (2,1))) / 3.
            if float(N) / 2 == round(N/2): # is even
                i_central = num_rep + 1
                W = np.concatenate((W[:i_central], W[i_central+1:]))
            u = W*u

        # Zero-padding

        # Field
        U = np.zeros(2*N-1, dtype=complex)
        U[0:N] = np.array(u)

        # Double frequency domain
        # Freq-space equivalent to doubling x-space domain
        fx = np.fft.fftfreq(2*N-1, self.d)

        # H = kernel in freq-space
        H = kernel(fx)

        if bandlimit:
            H = np.where(abs(fx) > bandlimit, 0, H)

        S = np.fft.ifft(np.fft.fft(U)*H)
        self.U = S[0:N]

    # -------------------------------------------------------------------

    # AUXILIARY METHODS: PROPAGATION CONDITIONS

    '''
        Bandwidth limit for far-field propagation using Angular Spectrum method

        u_limit = 1/[sqrt((2*df*z)^2+1)*wvl]

        Parameters:
            - z (double): propagation distance
            - df (double): sampling frequency / sampling interval in freq-space
    '''
    def _bandlimitAS(self, z, df):
        ulim = (2*df*z)**2 + 1
        ulim = self.wvl * np.sqrt(ulim)
        return 1/ulim
    

    # --------------------------------------------------------------------
    
    # PROPAGATION METHODS

    '''
        Propagation of optical waves based on the far-field Fraunhofer approximation.
        (1D)
        
        Parameters:
            - z (double): propagation distance
            
        Post:
            self.U = wave function after propagation
            self.x, self.d updated
            
        Notes:
            - Region of validity: z > 2D^2/lambda (lambda = wavelen, D = aperture diameter)
            - Valid near origin (x,y) ~ (0,0)
    '''
        
    def fraunhofer(self, z):
        N = len(self.U)
        k = 2*np.pi/self.wvl # optical wave vector
        fx = np.arange(-N//2, N//2)/(N*self.d)
        self.x = self.wvl*z*fx # observation plane coordinates
        self.U = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*(self.x**2))/(1j*self.wvl*z)*ft(self.U,self.d)
        self.d = self.wvl*z/(N*self.d)
        
        
    '''
        Propagation of optical waves based on the Fresnel approximation.
        Direct integration via fft, one-step integral method .
        (1D)
        
        Parameters:
            - z (double): propagation distance
            
        Post:
            - self.U = wave function after propagation
            - self.d = self.wvl*z/(N*self.d)
            - self.x updated with new d
            
        Notes:
            - N >= D1*wvl*z/(d*(wvl*z-D2*d))
            (D1 = source field maximum spatial extent)
            (D2 = observation field maximum spatial extent)
            
            - Output d is fixed.
              To define custom d, one must use the two-step integral method.

            - Suited only for far-field amplitude evaluation
        
    '''
    def fresnel_DI(self, z):
        N = len(self.U)
        k = 2*np.pi/self.wvl
        
        xin = self.x # source plane coordinates
        dout = self.wvl*z/(N*self.d)
        xout = np.arange(-N//2, N//2)*dout # observation plane coordinates
        
        A = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*(xout**2))/(1j*self.wvl*z)
        self.U = A*ft(self.U*np.exp(1j*k/(2*z)*(self.x**2)), self.d)
       
        self.d = dout
        self.x = xout


    '''
        Propagation of optical waves based on the Fresnel approximation
        Direct integration method via convolution 
        (1D)

        Parameters:
            - z (double): propagation distance

        Post: 
            - self.U = wave function after propagation

        Notes:
            - dout = din, xout = xin
            - Sampling condition: d <= wvl*z/L
    '''
    def fresnel_CV(self, z):
        def kernelFresnelCV(x):
            k = 2* np.pi / self.wvl
            H = np.exp(1j*k/(2*z)*x**2)
            A = np.exp(1j*k*z)/(1j*self.wvl*z)
            return A*H
        self._fft_conv_(kernelFresnelCV)

    '''
        Propagation of optical waves based on the Fresnel approximation
        Angular Spectrum method (convolution theorem)

        Parameters:
            - z (double): propagation distance
            - bandlimit (bool or double):
                if double, delete frequency components above given limit
                if True, calculate frequency limit and delete higher-frequency components
                if False, don't delete any frequency component
                True by default

        Post:
            - self.U = wave function after propagation

        Notes:
            - dout = din, xout = xin
            - Output field in arbitrary units, proper constant not applied
            - Sampling condition: d >= sqrt(wvl*z/N)
    '''
    def fresnel_AS(self, z, bandlimit=True):
        def kernelFresnelAS(fx):
            k = 2*np.pi/self.wvl
            fsq = fx**2
            H = np.sqrt(self.wvl*z)*np.exp(1j*np.pi/4)*np.exp(-1j*np.pi*self.wvl*z*fsq)        
            A = np.exp(1j*k*z)/(1j*self.wvl*z)
            return A*H

        # using same result as AS (approximately correct)
        if (type(bandlimit) is bool) and bandlimit: # bandlimit = True
            N = len(self.x)
            bandlimit = self._bandlimitAS(z,1/(2*self.d*N))
            
        self._ifft_conv_(kernelFresnelAS, bandlimit)

    '''
        Propagation of optical waves based on Rayleigh-Sommerfeld I formula.
        FFT convolution method.

        Parameters:
            - z (double): propagation distance
            - fast (bool): if True, use exponential insted of Hankle function for RS kernel
        
        Post:
            - self.U = wave function after propagation
            - Returns quality parameter (double)

        References:
            - https://dlmf.nist.gov/10.2#E5
			- F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.

        Notes:
            - This approach returns a quality parameter. If quality>1, propagation is correct.
            - Assuming n=1.
            - Result not normalized or scaled with proper constant.
            - dout = din, xout = xin
            - Sampling condition: d <= wvl*sqrt(z^2+L^2/2)/L
    '''
    def rayleigh_sommerfeld(self, z, fast=False):

        # Quality parameter
        dr_real = np.sqrt(self.d**2)
        rmax = np.sqrt((self.x**2).max())
        dr_ideal = np.sqrt(self.wvl**2 + rmax**2 + 2*self.wvl*np.sqrt(rmax**2+z**2)) - rmax
        quality = dr_ideal/dr_real

        def kernelRS(x):
            k = 2 * np.pi / self.wvl
            R = np.sqrt(x**2 + z**2)
            hk1 = None
            if fast:
                hk1 = np.sqrt(2/(np.pi*k*R)) * np.exp(1j*(k*R-3*np.pi/4))
            else:
                hk1 = hankel1(1, k*R)
            hk1 = hk1 * 0.5j * k * z / R
            return hk1

        self._fft_conv_(kernelRS)
        return quality

    '''
        Propagation of optical waves using angular spectrum representation
        Convolution method with analytical transfer function
        (1D)

        H = exp(1j*pi*z*m) where m = sqrt(1/wvl^2-fx^2)

        Parameters:
            - z (double): propagation distance
            - bandlimit (bool or double):
                if double, delete frequency components above given limit
                if True, calculate frequency limit and delete higher-frequency components
                if False, don't delete any frequency component
                True by default

        Post:
            - self.U = wave function after propagation

        Notes: 
            - dout = din, xout = xin
            - Sampling condition: wvl*z*N/(L*sqrt(L^2-2*(wvl*N/2)^2)) <= 1
            - Bandwidth condition: fx_limit = 1/(wvl*sqrt(1+(2*df*z)^2))
            - (bandwidth condition can be applied for good results in far field, even if sampling condition isn't met)
            - Notice that if zero-padding is used, then df'=2*df

        References:
            - K Matsushima, T Shimobaba. Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields.
    '''
    def angular_spectrum_repr(self, z, bandlimit=True):
        def kernelAS(fx):
            fsq = fx**2
            fsq = np.where(fsq>1/self.wvl**2, 0, fsq) # remove evanescent waves
            m = np.sqrt(1/self.wvl**2 - fsq)
            H = np.exp(1j*2*np.pi*z*m)
            return H
        
        if (type(bandlimit) is bool) and bandlimit: # bandlimit = True
            N = len(self.x)
            bandlimit = self._bandlimitAS(z,1/(2*self.d*N))
            
        self._ifft_conv_(kernelAS, bandlimit)

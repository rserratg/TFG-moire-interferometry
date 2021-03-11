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
            - kernel (function(x)): impulse response. Parameter x is x-space
            - simpson (bool): 
                if True, uses Simpson's rule for better acuracy
                True by default
                
        Note:
            - When using Simpson, it is highly advised to use N odd
        References:
            - Ji Qiang. A high-order fast method for computing convolution integral with smooth kernel.
    '''
    def _fft_conv(self, kernel, simpson=True):
        
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
            - Important: notice that the domain is doubled due to zero-padding.
              Therefore, df'=2*df
            - When using Simpson rule, it is highly advised to use N odd
    '''
    def _ifft_conv(self, kernel, bandlimit=False, simpson=True):
        
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
            H = np.where(np.abs(fx) > bandlimit, 0, H)

        S = np.fft.ifft(np.fft.fft(U)*H)
        self.U = S[0:N]

        # Print bandlimit and max freq for debugging
        #print("B: ", bandlimit)
        #print("f: ", fx.max())        

    '''
        Propagate field using impulse response.
        (Convolution fft calculation for propagation methods)
        Not using zero padding nor Simpson's rule.

        S = IFFT[FFT(U)*FFT(H)]

        Parameters:
            - kernel (function(x)): impulse response. Parameter x is x-space.
        
        Note:
            - Faster than using zero padding.
            - Might produce aliasing errors.
    '''
    def _fft_conv_nopad(self, kernel):
        
        N = len(self.x)
        x = self.x
        u = self.U

        H = ft(kernel(x))
        S = ift(ft(u)*H) * self.d
        self.U = S

    
    '''
        Propagate field using transfer function.
        (Convolution fft calculation for propagation methods)
        Not using zero-padding nor Simpson's rule.

        S = IFFT[FFT[U]*H]

        Parameters:
            - kernel (function(fx)): transfer function, parameter fx is freq-space.
            - bandlimit (False or double):
                if double, delete frequency components above given limit
                if False, don't delete any frequency component
                False by default
            
        Notes:
            - If bandlimit=0 it is treated as if it were False.
            - Faster than using zero-padding.
            - Might produce aliasing errors.
    '''
    def _ifft_conv_nopad(self, kernel, bandlimit=False):
            
            N = len(self.x)
            u = self.U
            
            fx = np.fft.fftfreq(N, self.d)
            H = kernel(fx)

            if bandlimit:
                H = np.where(np.abs(fx) > bandlimit, 0, H)

            S = np.fft.ifft(np.fft.fft(u)*H)
            self.U = S


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

    # AUXILIARY METHODS: PROPAGATION KERNELS

    '''
        Propagation kernel: Fresnel convolution

        Real space.
        Paraxial approximation.

        Parameters:
            - z (double): propagation distance
            - x (np.array): x-space
    '''
    def _kernelFresnelCV(self, z, x):
        k = 2* np.pi / self.wvl
        H = np.exp(1j*k/(2*z)*x**2)
        A = np.exp(1j*k*z)/np.sqrt(1j*self.wvl*z)
        return A*H


    '''
        Propagation kernel: Fresnel AS

        Frequency space.
        Paraxial approximation.

        Parameters:
            - z (double): propagation distance
            - fx (np.array): freq space
    '''
    def _kernelFresnelAS(self, z, fx):
        k = 2*np.pi/self.wvl
        fsq = fx**2
        H = np.exp(-1j*np.pi*self.wvl*z*fsq)        
        A = np.exp(1j*k*z)
        return A*H


    '''
        Propagation kernel: Rayleigh-Sommerfeld

        Real space.
        Assuming R >> wvl.

        Parameters:
            - z (double): propagation distance
            - fast (bool): if True, use exponential instead of Hankel function.
            - x (np.array): x-space
            
        Note:
            - scipy hankel1 function returns Nan for large values of k*R (around 1e10)
    '''
    def _kernelRS(self, z, fast, x):
        k = 2 * np.pi / self.wvl
        R = np.sqrt(x**2 + z**2)
        hk1 = None
        if fast:
            hk1 = np.sqrt(2/(np.pi*k*R)) * np.exp(1j*(k*R-3*np.pi/4))
        else:
            hk1 = hankel1(1, k*R)
        hk1 = hk1 * 0.5j * k * z / R
        return hk1


    '''
        Propagation kernel: Angular Spectrum representation

        Frequency space.

        Parameters:
            - z (double): propagation distance
            - fx (np.array): freq space
    '''
    def _kernelAS(self, z, fx):
        fsq = fx**2
        fsq = np.where(fsq>1/self.wvl**2, 0, fsq) # remove evanescent waves
        m = np.sqrt(1/self.wvl**2 - fsq)
        H = np.exp(1j*2*np.pi*z*m)
        return H

    # --------------------------------------------------------------------
    
    # AUXILIARY METHODS: GENERAL PROPAGATION FUNCTION

    '''
        General function for propagation using convolution method.
        
        Parameters:
            - method (string): propagation method - "AS", "RS", "FresnelAS", "FresnelCV"
            - z (double): propagation distance
            - pad (bool): if True, use zero-padding. True by default.
            - Conditional parameters:
                - bandlimit (bool or double):
                    Only if method="AS" or method="FresnelAS" (kernel in freq space).
                    if False, use all frequency components.
                    if double, use only frequency components up to given value.
                    if True, calculate higher frequency to be used.
                    True by default.
                - fast:
                    Only if method="RS".
                    if False, use exponential instead of Hankel function.
                - simpson:
                    Only if pad=True (zero-padding is used).
                    if True, use Simpson rule for improved accuracy
                    (note that it doesn't guarantee better results)
                    True by default.
    '''
    def _conv_propagation(self, z, method, pad=True, **kwargs):

        # If True, kernel in real space (RS, FresnelCV)
        # If False, kernel in frequency domain (AS, FresnelAS)
        kernelspace = True # True if RS/FresnelCV, False if AS/FresnelAS
        
        # Kernel function
        kernel = None
        if method == "AS":
            kernel = lambda fx : self._kernelAS(z,fx)
            kernelspace = False
        elif method == "RS":
            fast = kwargs.get("fast", False)
            kernel = lambda x : self._kernelRS(z,fast,x)
        elif method == "FresnelAS": 
            kernel = lambda fx : self._kernelFresnelAS(z,fx) 
            kernelspace = False
        elif method == "FresnelCV":
            kernel = lambda x : self._kernelFresnelCV(z,x)
        else:
            print("Propagation test: invalid kernel")
            return
            
        # Propagation
        if kernelspace:
            if pad:
                simpson = kwargs.get("simpson", True)
                self._fft_conv(kernel, simpson)
            else:
                self._fft_conv_nopad(kernel)
        else:
            # If bandlimit = True: calculate max freq
            bandlimit = kwargs.get("bandlimit", True)
            if (type(bandlimit) is bool) and bandlimit:
                N = len(self.x)
                df = 1/(self.d*N)
                if pad: # if zero-padding is used, domain is doubled
                    df /= 2
                bandlimit = self._bandlimitAS(z,df)
            
            if pad:
                simpson = kwargs.get("simpson", True)
                self._ifft_conv(kernel, bandlimit, simpson)
            else:
                self._ifft_conv_nopad(kernel, bandlimit)

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
        
        fx = np.fft.fftshift(np.fft.fftfreq(N, self.d))        
        self.x = self.wvl*z*fx # observation plane coordinates
        A = np.exp(1j*k*z)/np.sqrt(1j*self.wvl*z)
        self.U = A*np.exp(1j*k/(2*z)*(self.x**2))*ft(self.U,self.d)
        self.d = self.x[1] - self.x[0]
        
        
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
        xout = self.wvl*z*np.fft.fftshift(np.fft.fftfreq(N,self.d))
       
        
        A = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*(xout**2))/np.sqrt(1j*self.wvl*z)
        self.U = A*ft(self.U*np.exp(1j*k/(2*z)*(xin**2)), self.d)
       
        self.d = xout[1]-xout[0]
        self.x = xout


    '''
        Propagation of optical waves based on the Fresnel approximation
        Direct integration method via convolution 
        (1D)

        Parameters:
            - z (double): propagation distance
            - simpson (bool): if True, use Simpson rule. True by default.

        Post: 
            - self.U = wave function after propagation

        Notes:
            - dout = din, xout = xin
            - Sampling condition: d <= wvl*z/L
            - Zero padding is used.
    '''
    def fresnel_CV(self, z, pad=True, simpson=True):
        self._conv_propagation(z, "FresnelCV", simpson=simpson)


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
            - simpson (bool): if True, use simpson rule. True by default.

        Post:
            - self.U = wave function after propagation

        Notes:
            - dout = din, xout = xin
            - Sampling condition: d >= sqrt(wvl*z/N)
            - Zero padding is used
    '''
    def fresnel_AS(self, z, bandlimit=True, pad=True, simpson=True):
        self._conv_propagation(z, "FresnelAS", bandlimit=bandlimit, simpson=simpson)


    '''
        Propagation of optical waves based on Rayleigh-Sommerfeld I formula.
        FFT convolution method.

        Parameters:
            - z (double): propagation distance
            - fast (bool): if True, use exponential insted of Hankle function for RS kernel
            - simpson (bool): if True, use Simpson rule. True by default 
        
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
            - Zero padding is used
    '''
    def rayleigh_sommerfeld(self, z, fast=False, simpson=True):

        # Quality parameter
        dr_real = np.sqrt(self.d**2)
        rmax = np.sqrt((self.x**2).max())
        dr_ideal = np.sqrt(self.wvl**2 + rmax**2 + 2*self.wvl*np.sqrt(rmax**2+z**2)) - rmax
        dr_ideal /= 2
        quality = dr_ideal/dr_real

        self._conv_propagation(z, "RS", fast=fast, simpson=simpson)

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
            - simpson (bool): if True, use Simpson rule. True by default

        Post:
            - self.U = wave function after propagation

        Notes: 
            - dout = din, xout = xin
            - Sampling condition: wvl*z*N/(L*sqrt(L^2-2*(wvl*N/2)^2)) <= 1
            - Bandwidth condition: fx_limit = 1/(wvl*sqrt(1+(2*df*z)^2))
            - (bandwidth condition can be applied for good results in far field, even if sampling condition isn't met)
            - Zero padding is used. Notice that, in this case, df'=df/2 

        References:
            - K Matsushima, T Shimobaba. Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields.
    '''
    def angular_spectrum_repr(self, z, bandlimit=True, simpson=True):
        self._conv_propagation(z, "AS", bandlimit=bandlimit, simpson=simpson)

    '''
        Step for Beam Propagation Method (Fresnel approximation)
        Same conditions as AS/Fresnel-AS
        Only suited for near-field propagation (very small z)
        
        Parameters:
            - deltaz (double): propagation distance
            - n (double): refraction index
        
        Implementation taken from 'diffractio'
        References:
            - TC Poon, T Kim. Engineering optics with Matlab.
            - L M Sanchez Brea. Diffractio, python module for diffraction and interference optics. https://pypi.org/project/diffractio
    '''
    def bpm(self, deltaz, n=1):
        k0 = 2 * np.pi / self.wvl
    
        N = len(self.x)
        L = self.x[-1] - self.x[0]
        
        # Initial field
        field_z = self.U
        
        # Wave number in 1D (frequencies)
        kx1 = np.linspace(0, N / 2 - 1, N / 2)
        kx2 = np.linspace(-N / 2, -1, N / 2)
        kx = (2 * np.pi / L) * np.concatenate((kx1, kx2))
        
        # Normalized phase
        phase1 = np.exp((1j * deltaz * kx**2) / (2*k0))
        phase2 = np.exp(-1j * n * k0 * deltaz)
        
        # Edge filter (supergaussian function)
        pixelx = np.linspace(-N/2, N/2, N)
        edgeFilter = np.exp(-((pixelx) / (0.98 * 0.5 * N))**90)
    
        # Field propagation
        field_z = np.fft.ifft(np.fft.fft(field_z) * phase1) * phase2
        self.U = field_z * edgeFilter

# Propagation methods for the OptWave class

import numpy as np
from ._utils import ft, ift
import matplotlib.pyplot as plt
from scipy.special import hankel1

# FIXME: functions in proptest work properly
# Some of the regular functions do not
# Fix regular functions using proptest

'''
    Helper (Mixin) class for the OptWave class.
    
    Contains the methods related to wave propagation methods.
'''


class MixinProp:

    '''
        Propagation of optical waves based on the far-field Fraunhofer approximation.
        (1D)
        
        Parameters:
            - z: propagation distance
            
        Post:
            self.U = wave function after propagation
            self.x, self.d updated
            
        Notes:
            - Region of validity: z > 2D^2/lambda (lambda = wavelen, D = aperture diameter)
            - Valid near origin
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
        One-step integral method.
        (1D)
        
        Parameters:
            - z: propagation distance
            
        Post:
            - self.U = wave function after propagation
            - self.x updated
            - self.d = self.wvl*z/(N*self.d)
            
        Notes:
            - N >= D1*wvl*z/(d*(wvl*z-D2*d))
            (D1 = source field maximum spatial extent)
            (D2 = observation field maximum spatial extent)
            
            - Output d is fixed.
              To define custom d, use two-step propagation.
        
    '''
    def fresnel_integral_one_step(self, z):
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
        Angular spectrum method (convolution theorem)
        (1D)
        
        Parameters:
            - z: propagation distance
            
        Post:
            - self.U = wave function after propagation
            - self.d, self.x are NOT modified
            
        Notes:
            - Direct application of convolution theorem results in dout = din, xout = xin  
              To define custom d, must rewrite the integral adding scaling parameter.
            - Best suited for only short distances
            - N >= wvl*z/(d**2)
            - d <= wvl*z/(D1+D2)
            
            - Transfer function not working properly
              Results differ between both versions (calculated by hand vs dft)
    '''
    def fresnel_ang_spec(self, z):
        N = len(self.U)
        k = 2*np.pi/self.wvl
        
        df = 1/(self.d*N)
        fx = np.arange(-N//2, N//2)*df;
        fsq = fx**2
        
        # Transfer function
        H = np.sqrt(self.wvl*z)*np.exp(1j*np.pi/4)*np.exp(-1j*np.pi*self.wvl*z*fsq)        
        
        #H = ft(np.exp(1j*k/(2*z)*self.x**2), self.d)
        A = np.exp(1j*k*z)/(1j*self.wvl*z)
        
        self.U = A*ift(H*ft(self.U,self.d), df)
        
            

    '''
        Propagation of optical waves based on Rayleigh-Sommerfeld I formula.
        FFT convolution method.
        
        Parameters:
            - z (float): propagation distance
            - fast (bool): if True, use exponential instead of Hankel function for RS kernel
            
        Returns:
            (float) quality parameter
            
        References:
			- https://dlmf.nist.gov/10.2#E5
			- F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.
			- A high-order fast method for computing convolution integral with smooth kernel
            
        Notes: 
            - This approach returns a quality parameter. If quality>1, propagation is right
            - Assuming n=1
            - Result not normalized, constant might be incorrect
    '''
    def rayleigh_sommerfeld(self, z, fast=False):
        N = len(self.x)
        x = self.x
        U = self.U
        
        # Quality parameter
        dr_real = np.sqrt(self.d**2)
        rmax = np.sqrt((self.x**2).max())
        dr_ideal = np.sqrt(self.wvl**2 + rmax**2 + 2*self.wvl*np.sqrt(rmax**2+z**2)) - rmax
        quality = dr_ideal/dr_real
        
        # Simpson integration rule
        a = [2,4]
        num_rep = int(round(N/2)-1)
        b = np.array(a * num_rep)
        W = np.concatenate(((1,), b, (2,1))) / 3.
        
        if float(N) / 2 == round(N/2): # is even
            i_central = num_rep + 1
            W = np.concatenate((W[:i_central], W[i_central+1:]))
            
        # field
        U = np.zeros(2*N-1, dtype=complex)
        U[0:N] = np.array(W*self.U)
        
        # Double domain
        xext = self.x[0] - x[::-1]
        xext = xext[0:-1]
        x = np.concatenate((xext,self.x-x[0]))
            
        # H = kernelRS
        k = 2 * np.pi / self.wvl
        R = np.sqrt(x**2 + z**2)
        hk1 = None
        if fast:
            hk1 = np.sqrt(2/(np.pi*k*R)) * np.exp(1j*(k*R-3*np.pi/4))
        else:
            hk1 = hankel1(1, k*R)
        hk1 = hk1 * 0.5j * k * z / R
        
        S = np.fft.ifft(np.fft.fft(U)*np.fft.fft(hk1))*self.d
        self.U = S[N-1:]
        
        return quality
        
        
    '''
        Propagation using angular spectrum representation transfer function
        
        H = exp(1j*pi*z*m) where m = sqrt(1/wvl^2-fx^2)
        
        Notes:
            - Result fluctuates a lot (similar to fresnel_ang_spec transfer function)
            - I can't make this work with Simpson rule
    '''
    def ang_spec_repr(self, z):
        N = len(self.U)
        k = 2*np.pi/self.wvl
        
        df = 1/(self.d*N)
        fx = np.arange(-N//2, N//2)*df;
        fsq = fx**2
        
        fsq = np.where(fsq>1/self.wvl**2, 0, fsq) # remove evanescent waves
        m = np.sqrt(1/self.wvl**2-fsq)
        H = np.exp(1j*2*np.pi*z*m)
            
        self.U = ift(H*ft(self.U,self.d), df)
        
        
# --------------------------------------------------------------------------
#  TEST FUNCTIONS        

    '''
        Convolution fft calculation for propagation methods
        (not using Simpson rule, but only zero-padding)
        
        Parameters:
            - kernel (function(x)): impulse response, parameter x is x-space
   '''
    
    def _fft_conv_(self, kernel):
        
        N = len(self.x)
        x = self.x
        U = self.U
        
        #Zero-padding
        
        # field
        U = np.zeros(2*N-1, dtype=complex)
        U[0:N] = np.array(self.U)
        
        # Double domain
        xext = self.x[0] - x[::-1]
        xext = xext[0:-1]
        x = np.concatenate((xext,self.x-x[0]))
       
        
        # H = kernel in freq-space
        H = np.fft.fft(kernel(x))
        
        S = np.fft.ifft(np.fft.fft(U)*H)*self.d
        self.U = S[N-1:]
        
        
    '''
        Convolution fft calculation for propagation methods
        
        Parameters:
            - kernel (function(fx)): transfer function, parameter fx in freq-space
    '''
    def _ifft_conv_(self, kernel):
        N = len(self.U)
        k = 2*np.pi/self.wvl
        
        
        '''
        # No zero-padding
        
        df = 1/(self.d*N)
        fx = np.arange(-N//2, N//2)*df;
        fx = np.fft.ifftshift(fx) # Put zero frequency at index 0
        
        H = kernel(fx)
        H = np.where(abs(fx)>2e4, 0, H) # bandwidth limitation
        
        self.U = np.fft.ifft(H*np.fft.fft(self.U))
        
        return
        '''
        
        # Zero padding
        U = np.zeros(2*N-1, dtype=complex)
        U[0:N] = np.array(self.U)
        
        fx = np.fft.fftfreq(2*N-1, self.d)
        
        H = kernel(fx)
        #H = np.where(abs(fx)>4e4, 0, H)   
        
        S = np.fft.ifft(np.fft.fft(U)*H)
        self.U = S[0:N]
                
    
    '''
        Function to test fft_convolution with different kernel functions
        
        Parameters:
            - z: propagation distance
            - method: propagation method
                      RS, AS, FresnelCV, FresnelAS
    '''
    def test_prop(self, z, method='RS'):
        
        def kernelRS(x, fast=False):
            # H = kernelRS
            k = 2 * np.pi / self.wvl
            R = np.sqrt(x**2 + z**2)
            hk1 = None
            
            if fast:
                hk1 = np.sqrt(2/(np.pi*k*R)) * np.exp(1j*(k*R-3*np.pi/4))
            else:
                hk1 = hankel1(1, k*R)
            hk1 = hk1 * 0.5j * k * z / R
            return hk1
            
        def kernelAS(fx):
            fsq = fx**2
            fsq = np.where(fsq>1/self.wvl**2, 0, fsq) # remove evanescent waves
            m = np.sqrt(1/self.wvl**2 - fsq)
            H = np.exp(1j*2*np.pi*z*m)
            return H
            
        def kernelFresnel(x):
            k = 2* np.pi / self.wvl
            H = np.exp(1j*k/(2*z)*x**2)
            A = np.exp(1j*k*z)/(1j*self.wvl*z)
            return A*H
            
        def kernelFresnelAS(fx):
            k = 2*np.pi/self.wvl
            fsq = fx**2
            H = np.sqrt(self.wvl*z)*np.exp(1j*np.pi/4)*np.exp(-1j*np.pi*self.wvl*z*fsq)        
            A = np.exp(1j*k*z)/(1j*self.wvl*z)
            return A*H
            
        if method=='RS':
            self._fft_conv_(kernelRS)
        elif method=='AS':
            self._ifft_conv_(kernelAS)
        elif method=='FresnelCV':
            self._fft_conv_(kernelFresnel)
        elif method=='FresnelAS':
            self._ifft_conv_(kernelFresnelAS)
        else:
            print("Test_prop: invalid propagation method")
            


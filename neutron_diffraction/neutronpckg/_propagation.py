import numpy as np
from scipy.constants import hbar, m_n

'''
    Helper (Mixin) class for the NeutronWave class
    
    Contains methods related to propagation of the neutron's wavefunction
    
    Note:
        - The global coordinates of the spaces (self.X) have to be updated accordingly after propagation
'''

class MixinProp:

    # AUXILIARY METHODS: CONVOLUTION CALCULATIONS

    '''
        Propagate field using transfer function
        (Convolution fft calculation for propagation methods)
        Using zero-padding

        S = IFFT[FFT(U)*H]

        Parameters:
            - kernel(function(kx)): transfer function, parameter ks is momentum-space
            - bandlimit (False or double):
                if double, delete frequency components above given limit
                if False, don't delete any frequency components
                False by default
        
        Post:
            - self.Psi = wavefunction after propagation

        Notes:
            - if bandlimit=0 it is treated as if it were False.
            - Important: notice that the domain is double due to zero-padding.
              Therefore, dk'=2*dk
    '''
    def _ifft_conv(self, kernel, bandlimit=False):

        k = 2*np.pi/self.wvl

        # Zero-padding

        # Wavefunction
        Psi = np.zeros((self.Nn, 2*self.Nx-1), dtype=np.complex128)
        Psi[:, 0:self.Nx] = self.Psi

        # Double frequency domain
        kx = 2*np.pi*np.fft.fftfreq(2*self.Nx-1, self.d)

        # H = kernel in freq-space
        H = kernel(kx)

        # if bandlimit != False, apply bandlimit (assuming bandlimit != 0)
        if bandlimit:
            H = np.where(np.abs(kx) > bandlimit, 0, H)

        # Multiply each row of fft(Psi) by kernel
        S = np.fft.ifft(np.fft.fft(Psi,axis=1) * H, axis=1)
        self.Psi = S[:, 0:self.Nx]


    '''
        Propagate field using transfer function
        (Convolution fft calculation for propagation methods)
        Not using zero-padding

        S = IFFT[FFT(U)*H]

        Parameters:
            - kernel(function(kx)): transfer function, parameter ks is momentum-space
            - bandlimit (False or double):
                if double, delete frequency components above given limit
                if False, don't delete any frequency components
                False by default
        
        Post:
            - self.Psi = wavefunction after propagation

        Notes:
            - if bandlimit=0 it is treated as if it were False.
    '''
    def _ifft_conv_nopad(self, kernel, bandlimit=False):

        k = 2*np.pi/self.wvl

        # Frequencies (momentum)
        kx = 2*np.pi*np.fft.fftfreq(self.Nx, self.d)

        # H = kernel in freq-space
        H = kernel(kx)

        # if bandlimit != False, apply bandlimit (assuming bandlimit != 0)
        if bandlimit:
            H = np.where(np.abs(kx) > bandlimit, 0, H)

        # Multiply each row of fft(Psi) by kernel
        S = np.fft.ifft(np.fft.fft(self.Psi,axis=1) * H, axis=1)
        self.Psi = S


    # --------------------------------------------------------------------

    # AUXILIARY METHODS: PROPAGATION CONDITIONS

    '''
        Bandwidth limit for far-field propagation using Fresnel-AS method

        u_limit = k*pi/(dk*z)

        Parameters:
            - z (double): propagation distance
            - dk (double): smapling interval in momentum space
    '''
    def _bandlimitFresnelAS(self, z, dk):
        k = 2*np.pi/self.wvl
        return k*np.pi/(z*dk)


    # --------------------------------------------------------------------

    # AUXILIARY METHODS: PROPAGATION KERNELS

    '''
        Propagation kernel: Fresnel AS

        Momentum space.
        Paraxial approximation.

        Parameters:
            - z (double): propagation distance
            - kx (np.array): momentum space
    '''
    def _kernelFresnelAS(self, z, kx):
        k = 2*np.pi/self.wvl 
        ksq = kx**2
        H = np.exp(-1j*ksq*z/(2*k))
        return H

    
    # ------------------------------------------------------------------

    # PROPAGATION METHODS

    '''
        Standard propagation method.
        Propagation based on Fresnel approximation.
        Angular Spectrum method (convolution theorem)

        Parameters: 
            - z (double): propagation distance
            - bandlimit (bool or double): 
                if double, delete momentum components above given limit
                if True, calculate momentum limit and delete higher components
                if False, don't delete any momentum component
                By default True.
            - pad (bool): If True, use-zero padding. By default True.
        
        Post:
            - self.Psi = wave function after propagation
            - self.L = total propagation distance
            - self.X = output global coordinates
    '''
    def propagate(self, z, bandlimit=True, pad=True):

        # Get kernel as function of kx
        kernel = lambda kx : self._kernelFresnelAS(z, kx)
        
        # If bandlimit == True, calculate bandlimit
        if (type(bandlimit) is bool) and bandlimit:
            df = 2*np.pi/(self.d * self.Nx) 
            if pad:
                df /= 2
            bandlimit = self._bandlimitFresnelAS(z, df)

        # Update Psi
        if pad:
            self._ifft_conv(kernel, bandlimit=bandlimit)
        else:
            self._ifft_conv_nopad(kernel, bandlimit=bandlimit)

        # Update propagation distance and global coordinates
        self.L += z
        self.X[:] += z*np.tan(self.theta[:])

    '''
        Propagate with linar potential
        Paraxial approximation
        Using zero-padding

        Potential: V(x) = F*x 

        Parameters:
            - z (double): propagation distance
            - F (double): potential coefficient
            - bandlimit (bool or double): 
                if False use all momentum components in transfer function.
                if double limit maximum momentum to given value
                if True calculate maximum momentum
                By default True.
            - pad (bool): if True, use zero padding. By default True.

        Post:
            - self.Psi = wave function after propagation
            - self.L = total propagation distance
            - self.X = output global coordinates
            - self.theta = output beam angle (related to global linear phase)
    '''
    def propagate_linear_potential(self, z, F, bandlimit=True, pad=True):

        k = 2*np.pi/self.wvl

        # Propagate with Fresnel
        self.propagate(z, bandlimit, pad)

        # Apply field displacement due to potential
        self.X[:] -= (z**2*F*m_n)/(hbar*k*np.cos(self.theta))**2/2
        
        # Apply linear phase
        self.theta = np.arctan(np.tan(self.theta) - (F*z*m_n)/(hbar*k*np.cos(self.theta)**2))
import numpy as np
from scipy.constants import hbar, m_n

'''
    Helper (Mixin) class for the NeutronWave class
    
    Contains methods related to propagation of the neutron's wavefunction
    
    Note:
        - The global coordinates of the spaces (self.X) have to be updated accordingly after propagation
'''

class MixinProp:

    '''
        Propagation in paraxial approximation
        
        Parameters:
            - z (double): propagation distance
            - bandlimit (double): 
                if False, all momentum components are used
                if double, momentum components above given value are eliminated
                if True, maximum momentum is calculated from Nyquist theorem
                By default True.
    '''
    def propagate(self, z, bandlimit=True):
    
        k = 2 * np.pi / self.wvl

        # Zero-padding
        
        # Wavefunction
        Psi = np.zeros((self.Nn, 2*self.Nx-1), dtype=np.complex128)
        Psi[:, 0:self.Nx] = self.Psi        
        
        # Double frequency domain
        kx = 2*np.pi*np.fft.fftfreq(2*self.Nx-1, self.d)
        
        # H = kernel in freq-space
        H = np.exp(-1j * kx**2 * z/(2*k))
        
        # if bandlimit == True, calculate max momentum
        if (type(bandlimit) is bool) and bandlimit:  
            df = 2*np.pi/(2 * self.Nx * self.d)
            bandlimit = k * np.pi / (z*df)
           
        # if bandlimit != False, apply bandlimit (assuming bandlimit != 0)
        if bandlimit:
            H = np.where(np.abs(kx) > bandlimit, 0, H)
            
        # Multiply each row of fft(Psi) by kernel
        S = np.fft.ifft(np.fft.fft(Psi, axis=1) * H, axis=1)
        self.Psi = S[:, 0:self.Nx]
        
        # Update propagation distance
        self.L += z
        
        # Update global coordinates
        self.X[:] += z*np.tan(self.theta[:])
        
        
    # WARNING: TEST FUNCTION
    '''
        Propagation in paraxial approximation without using zero-padding
        
        Parameters:
            - z (double): propagation distance
            - bandlimit (double): 
                if False, all momentum components are used
                if double, momentum components above given value are eliminated
                if True, maximum momentum is calculated from Nyquist theorem
                By default True.
    '''
    def propagate_nopad(self, z, bandlimit=True):
    
        k = 2 * np.pi / self.wvl

        # Frequencies (momentum)
        kx = 2*np.pi*np.fft.fftfreq(self.Nx, self.d)
        
        # H = kernel in freq-space
        H = np.exp(-1j * kx**2 * z/(2*k))
        
        # if bandlimit == True, calculate max momentum
        if (type(bandlimit) is bool) and bandlimit:  
            df = 2*np.pi/(self.Nx * self.d)
            bandlimit = k * np.pi / (z*df)
           
        # if bandlimit != False, apply bandlimit (assuming bandlimit != 0)
        if bandlimit:
            H = np.where(np.abs(kx) > bandlimit, 0, H)
            
        # Multiply each row of fft(Psi) by kernel
        S = np.fft.ifft(np.fft.fft(self.Psi, axis=1) * H, axis=1)
        self.Psi = S
        
        # Update propagation distance
        self.L += z
        
        # Update global coordinates
        self.X[:] += z*np.tan(self.theta[:])

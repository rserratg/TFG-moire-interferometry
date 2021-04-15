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
            
        TODO: CHANGE TO ACCOUNT FOR LOCALITY OF NEUTRONS SPACE
        TODO: ZERO-PADDING OR NO ZERO-PADDING?
        TODO: BANDLIMIT?
        TODO: CHECK SHAPE OF RESULTING PSI
        TODO: CHECK UPDATE OF L AND X
    '''
    def propagate(self, z):
    
        # kernel
        def kernel(kx):
            k = 2*np.pi/self.wvl
            #vg = hbar*k/m_n # group velocity
            #H = np.exp(-1j*hbar*kx**2*z/(2*m_n*vz))   
            H = np.exp(-1j * kx**2 * z/(2*k))    
            return H
            
        # Zero-padding
        
        # Wavefunction
        Psi = np.zeros((self.Nn, 2*self.Nx-1), dtype=np.complex128)
        Psi[:, 0:self.Nx] = self.Psi        
        
        # Double frequency domain
        kx = 2*np.pi*np.fft.fftfreq(2*self.Nx-1, self.d)    
        
        # H = kernel in freq-space
        H = kernel(kx)
        
        # if bandlimit:     
        #   H = np.where(np.abs(kx) > bandlimit, 0, H)
        
        # Multiply each row of fft(Psi) by kernel
        S = np.fft.ifft(np.fft.fft(Psi, axis=1) * H, axis=1)
        self.Psi = S[:, 0:self.Nx]
        
        # Update propagation distance
        self.L += z
        
        # Update global coordinates
        self.X[:] += z*np.tan(self.theta[:])

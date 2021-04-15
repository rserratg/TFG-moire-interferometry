import numpy as np

'''
    Helper (Mixin) class for the NeutronWave class
    
    Contains methods related to initialization of neutron wavefunctions.
'''

class MixinWave:

    '''
        Slit source.
        
        Neutron beam considering coherence length at a distance L after a slit.
        Assuming slit is far from source.
        
        Parameters:
            - L0: distance from slit
            - sw (double): slit width [m]. By default 1e-3.
            - theta (double): divergence angle of neutron beam [radians]. By default 3.5e-3.
            
        Note:
            - If Nn = 1 the neutron is placed at the center of the slit (x0=0).
              Otherwise they are uniformly distributed over the slit (including edges)
    '''
    def slitSource(self, L0, sw=1e-3, theta=3.5e-3):
        
        # Place neutrons uniformly over slit
        if self.Nn == 1:
            self.x0 = np.array([[0.]])
        else:
            self.x0 = np.round(np.linspace(-sw/2, sw/2, self.Nn)/self.d)*self.d
            self.x0 = self.x0[np.newaxis].transpose()
            
        # Divergence angle of each neutron
        self.theta = theta*(2*np.random.rand(self.Nn,1) - 1)
              
        # Initial propagation distance
        self.L = L0
        
        # Update sampling space
        self.X[:] += self.x0[:] + L0*self.theta[:]
        
        # Coherence length
        sigma = self.wvl * L0 / sw
        
        # Wavefunction after slit source (important to keep complex data type)
        self.Psi[:] = 1/(2*np.pi*sigma**2)**(1/4) * np.exp(-(self.x)**2/(4*sigma**2))

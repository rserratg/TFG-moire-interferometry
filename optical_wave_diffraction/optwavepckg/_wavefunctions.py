# Wavefunction initializers for the OptWave class

import numpy as np

'''
    Helper (Mixin) class for the OptWave class.
    
    Contains methods related to generation of wavefunctions.
'''

class MixinWave:

    '''
        Uniform plane wave
        
        Parameters:
            - A: amplitude of the wave (A=1 by default)
        
        Post:
            Returns plane wave of amplitude 1
            (Output array dtype is complex)
        
    '''
    def planeWave(self, A=1):
        self.U.fill(A)

import neutron_test as n
import numpy as np

# Prepare and experiment, setting number of neutrons, slit size,
#beam divergence, and coherence length
twoG=n.Experiment(N_neutrons=1, slit=500E-6, theta=0)

# Initial propogation distance from slit to first object.
#This simply increases the coherence length and prepares the
#neutron wavepackets
twoG.initialPropagation(1.7)

# Apply a square phase grating of 15um period and pi/2 phase strength
twoG.phaseGrating(phi=np.pi/2, P=2.4E-6)

twoG.propagate(6)

# You should see the diffraction pattern from a square grating
twoG.plotWave()

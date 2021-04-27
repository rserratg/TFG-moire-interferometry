# Example/test with 1 neutron.
# Single / double slit.
# Checked with results in Sanz - Neutron Matter-Wave Diffraction: A Computational Perspective.

import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave

Nx = 9e4
Sx = 2e-3
L0 = 1.2
sw = 281e-6
D = 23e-6
z = 5

wave = NeutronWave(Nx, Sx, Nn=1, wvl=18.45e-10)

# Plane wave of unit amplitude
wave.Psi = np.ones_like(wave.Psi)

#wave.slit(D)
wave.doubleSlit(126.3e-6, 22.2e-6)

wave.propagate(2)

x = wave.X.T
I = np.abs(wave.Psi.T)**2

plt.plot(x*1e6, I)
#plt.xlim(-200e-6, 200e-6)
plt.xlabel(r"x [$\mu$m]")
plt.ylabel(r"$|\Psi|^2$")
plt.show()

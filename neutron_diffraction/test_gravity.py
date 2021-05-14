import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from scipy import constants

Nx = 5e4
Sx = 5e-3
L0 = 1.2
sw = 200e-6
wvl = 0.44e-9

zvals = np.linspace(0, 5, 101)

g = constants.g
mn = constants.m_n

#Plotting settings
datamin = -1e-3
datamax = 1e-3
numbins = int(1e4)

I = []

wave0 = NeutronWave(Nx, Sx, Nn=1, wvl=wvl)
wave0.slitSource(L0, sw, theta=0, randPos=False)

center, hist0 = wave0.hist_intensity(numbins, xlimits=(datamin,datamax), retcenter=True)
I.append(hist0)

for z in zvals[1:]:
    
    print(z)

    wave = NeutronWave(Nx, Sx, Nn=1, wvl=wvl)
    wave.slitSource(L0, sw, theta=0, randPos=False)
    wave.propagate_linear_potential(z, mn*g)
    
    _, hist = wave.hist_intensity(numbins, xlimits=(datamin,datamax))
    I.append(hist)
    
    #plt.plot(center*1e6, hist)
    #plt.xlabel(r"x [$\mu$m]")
    #plt.ylabel(r"$|\Psi|^2$")
    #plt.show()
    
    
I = np.asanyarray(I)

plt.pcolormesh(center*1e6, zvals, I, shading='nearest')
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("z [m]")
clb = plt.colorbar()
clb.set_label(r"$|\Psi|^2$")
plt.tight_layout()
plt.show()

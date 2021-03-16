# Script to generate plots for Journal - propagation methods
# Plane wave w/ rectangular aperture

# Tests:
#  - Fraunhofer
#  - Fresnel DI
#  - RS / Fresnel-CV (w/o zero-padding)
#  - RS w/ zero-padding (w/o simpson)
#  - RS w/ zero-padding & simpson
#  - AS / Fresnel-AS (w/o zero-padding & bandlimit)
#  - AS w/ bandlimit (w/o zero-padding)
#  - AS w/ zero-padding (w/o bandlimit)
#  - AS w/ zero-padding & bandlimit

# To hide a plot, simply comment its "plt.show()" line

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave

# Sim parameters
N = 1000 # number of grid points
L = 4e-3 # total size of the grid [m]
wvl = 1e-6 # optical wavelength [m]
D = 0.5e-3 # aperture width [m]
z = .1 # propagation distance [m]


# FRAUNHOFER

wave1 = OptWave(N,L,wvl)
wave1.planeWave()
wave1.rectAperture(D)
wave1.fraunhofer(z)
plt.figure()
plt.plot(wave1.x, np.abs(wave1.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

# FRESNEL DI

wave2 = OptWave(N,L,wvl)
wave2.planeWave()
wave2.rectAperture(D)
wave2.fresnel_DI(z)
plt.figure()
plt.plot(wave2.x, np.abs(wave2.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

# RS w/o zero-padding

wave3 = OptWave(N,L,wvl)
wave3.planeWave()
wave3.rectAperture(D)
wave3._conv_propagation(z, "RS", pad=False)
plt.figure()
plt.plot(wave3.x, np.abs(wave3.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

# RS w/ zero-padding w/o simpson

wave4 = OptWave(N,L,wvl)
wave4.planeWave()
wave4.rectAperture(D)
wave4._conv_propagation(z, "RS", pad=True, simpson=False)
plt.figure()
plt.plot(wave4.x, np.abs(wave4.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

# RS w/ zero-padding & simpson

wave5 = OptWave(N,L,wvl)
wave5.planeWave()
wave5.rectAperture(D)
wave5._conv_propagation(z, "RS", pad=True, simpson=True)
plt.figure()
plt.plot(wave5.x, np.abs(wave5.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

# AS w/o zero-padding & bandlimit

wave6 = OptWave(N,L,wvl)
wave6.planeWave()
wave6.rectAperture(D)
wave6._conv_propagation(z, "AS", pad=False, bandlimit=False)
plt.figure()
plt.plot(wave6.x, np.abs(wave6.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

# AS w/ bandlimit w/o zero-padding

wave7 = OptWave(N,L,wvl)
wave7.planeWave()
wave7.rectAperture(D)
wave7._conv_propagation(z, "AS", pad=False, bandlimit=True)
plt.figure()
plt.plot(wave7.x, np.abs(wave7.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

# AS w/ zero-padding w/o bandlimit

wave8 = OptWave(N,L,wvl)
wave8.planeWave()
wave8.rectAperture(D)
wave8._conv_propagation(z, "AS", pad=True, bandlimit=False, simpson=True)
plt.figure()
plt.plot(wave8.x, np.abs(wave8.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()


# AS w/ zero-padding & bandlimit

wave9 = OptWave(N,L,wvl)
wave9.planeWave()
wave9.rectAperture(D)
wave9._conv_propagation(z, "FresnelAS", pad=True, bandlimit=True, simpson=True)
plt.figure()
plt.plot(wave9.x, np.abs(wave9.U))
plt.xlabel("x [m]")
plt.ylabel("Amplitude [arbitrary units]")
plt.xlim(-2e-3, 2e-3)
plt.show()
plt.close()

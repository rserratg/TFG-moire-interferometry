# Script to generate plots for Journal - diffraction of gaussian beam

# List of plots
#  - Amplitude gr, nu=1
#  - Amplitude gr, nu=1, enhanced, AS + Fresnel
#  - Amplitude gr, nu=2
#  - Amplitude gr, z=10mm
#  - Phase gr, pi/2, nu=3/2
#  - Phase gr, pi/2, z=10mm
#  - Phase gr, pi, z=5mm

# Plots with nu defined: near-field (self-imaging effect)
# Plots with distance defined: far-field (diffraction orders)

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import normalizedIntensity

# Sim parameters
N = 10000       # number of grid points
L = 2e-3        # grid size
wvl = 623.8e-9  # wavelength
P = 15e-6       # grating perid 
ff = 0.5        # grating duty cycle

w0 = 150e-6             # beam waist radius
zr = w0**2*np.pi/wvl    # rayleigh range

# Self-image distance for gaussian beam
# Assuming object placed at beam waist plane
# z0 = 0 -> R = inf, w = w0
# nu = order of distance
def image_distance(nu):
    aux = nu*P**2/wvl
    z = (1-np.sqrt(1-4*aux**2/zr**2))*zr**2/aux/2
    return z
    
# AMPLITUDE GR - nu=1
wave1 = OptWave(N,L,wvl)
wave1.gaussianBeam(w0=w0)
wave1.rectAmplitudeGrating(P)
wave1.angular_spectrum_repr(image_distance(1))
plt.figure()
plt.plot(wave1.x*1e6, normalizedIntensity(wave1.U))
plt.xlim(-300, 300)
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()
plt.close()


# AMPLITUDE GR - nu=1 - enchanced, AS+Fresnel
wave2 = OptWave(N,L,wvl)
wave2.gaussianBeam(w0=w0)
wave2.rectAmplitudeGrating(P)
wave2.angular_spectrum_repr(image_distance(1))
wave2bis = OptWave(N,L,wvl)
wave2bis.gaussianBeam(w0=w0)
wave2bis.rectAmplitudeGrating(P)
wave2bis.fresnel_AS(image_distance(1))
plt.figure()
plt.plot(wave2.x*1e6, normalizedIntensity(wave2.U))
plt.plot(wave2bis.x*1e6, normalizedIntensity(wave2bis.U), "--")
plt.xlim(-75, 75)
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()
plt.close()

# AMPLITUDE GR - nu=2
wave3 = OptWave(N,L,wvl)
wave3.gaussianBeam(w0=w0)
wave3.rectAmplitudeGrating(P)
wave3.angular_spectrum_repr(image_distance(2))
plt.figure()
plt.plot(wave3.x*1e6, normalizedIntensity(wave3.U))
plt.xlim(-300, 300)
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()
plt.close()


# AMPLITUDE GR - z=10mm
wave4 = OptWave(N,L,wvl)
wave4.gaussianBeam(w0=w0)
wave4.rectAmplitudeGrating(P)
wave4.angular_spectrum_repr(10e-3)
plt.figure()
plt.plot(wave4.x*1e6, normalizedIntensity(wave4.U))
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()
plt.close()


# PHASE GR - pi/2 - nu=3/2
wave5 = OptWave(N,L,wvl)
wave5.gaussianBeam(w0=w0)
wave5.rectPhaseGrating(P, np.pi/2)
wave5.angular_spectrum_repr(image_distance(3/2))
plt.figure()
plt.plot(wave5.x*1e6, normalizedIntensity(wave5.U))
plt.xlim(-300, 300)
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()
plt.close()


# PHASE GR - pi/2 - z=10mm
wave6 = OptWave(N,L,wvl)
wave6.gaussianBeam(w0=w0)
wave6.rectPhaseGrating(P, np.pi/2)
wave6.angular_spectrum_repr(10e-3)
plt.figure()
plt.plot(wave6.x*1e6, normalizedIntensity(wave6.U))
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()
plt.close()


# PHASE GR - pi - z=5mm
wave7 = OptWave(N,L,wvl)
wave7.gaussianBeam(w0=w0)
wave7.rectPhaseGrating(P, np.pi)
wave7.angular_spectrum_repr(5e-3)
plt.figure()
plt.plot(wave7.x*1e6, normalizedIntensity(wave7.U))
plt.xlabel(r"x [$\mu$m]")
plt.ylabel("Intensity [arbitrary units]")
plt.show()
plt.close()

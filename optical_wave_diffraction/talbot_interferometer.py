# Talbot interferometer
# Phase object imaging
# Setup from Ibarra 1992: "Talbot interferometry: a new geometry"

import numpy as np
import matplotlib.pyplot as plt
from optwavepckg import OptWave
from optwavepckg._utils import intensity, visibility, binning
   

N = 5e5 + 1
L = 10e-2
wvl = 0.6328e-6

P = 3.175e-4
zt = 2*P**2/wvl # 31.86cm

f = 25e-2 # focal length of imaging lens
D = 0.05e-2 # width of spatial filter (slit)

wave = OptWave(N,L,wvl)
wave.planeWave()
wave.rectAmplitudeGrating(P,x0=0)
wave.angular_spectrum_repr(zt/4)
# object
#wave.trapezoidPhaseObject(np.pi, 1e-2, 1.01e-2)
wave.lens(12.54/4)
wave.angular_spectrum_repr(zt/4)
wave.rectAmplitudeGrating(P)
wave.angular_spectrum_repr(2*f-zt/4)
wave.lens(f)
wave.angular_spectrum_repr(f)
#wave.rectAperture(D)
wave.angular_spectrum_repr(f)
    
x = wave.x
I = intensity(wave.U)

#print(visibility(I,x,P))

#plt.plot(x, np.angle(wave.U))
plt.plot(x, I)
plt.show()

#print(zt)

# Peak integral
#p1 = I[(x>-0.014) & (x<-0.008)].sum()
#p2 = I[(x>0.008) & (x<0.014)].sum()

#print(p1)
#print(p2)

# Binned plot
newI, newX = binning(I,x,1000)
plt.bar(newX, newI, width=newX[1]-newX[0],align='edge')
#plt.show()

'''
    Lens test 
    f' = (1/n)*12.54 m
    2n = number of fringes
    Fringes are only seen if no filter is applied!
'''

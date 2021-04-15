import numpy as np
import matplotlib.pyplot as plt
from neutronpckg import NeutronWave
from scipy.optimize import curve_fit

'''
Nx = 5e5
Sx = 10e-3
L0 = 1.2
sw = 281e-6

# TODO: test diffraction with theta != 0 compared to optical waves

wave = NeutronWave(Nx, Sx, Nn=3)

wave.slitSource(L0, sw, theta=0.5)

#wave.slit(5e-6)
#wave.doubleSlit(140e-6, 5e-6)
wave.rectPhaseGrating(2.4e-6, np.pi/2)
wave.propagate(0.2)

x = wave.X.T
I = np.abs(wave.Psi.T)**2

plt.plot(x, I)
#plt.xlim(-200e-6, 200e-6)
plt.show()
'''

def fitSin(x, u, P, xlim = None):

    xaux = x
    if xlim is not None:
        xmin, xmax = xlim
        cond1 = x >= xmin
        cond2 = x <= xmax
        cond = cond1 & cond2
        xaux = x[cond]
        u = u[cond]
    
    # Sin function to fit
    def fun(xx, a, b, c):
        return a + b*np.sin(2*np.pi*xx/P + c)
    
    # Fit function to data and retrieve optimal parameters
    popt, _ = curve_fit(fun, xaux, u)
    A, B, phi = popt
    
    # Calculate contrast
    C = abs(B/A)
    fit = fun(x, A, B, phi)
    return C, fit


Nx = 5e4
Sx = 10e-3
sw = 281e-6
L1 = 1.2
L = 2.99
D = 10e-3
L2 = L - L1 - D
P = 2.4e-6

print('Starting sim')
wave = NeutronWave(Nx, Sx, Nn=500)

print(f'Sampling: {wave.d*1e6} um')

print('Slit source')
wave.slitSource(L1, sw, theta=0.5*np.pi/180)

print('G1 and first propagation')
wave.rectPhaseGrating(P, np.pi/2)
wave.propagate(D)

print('G2 and second propagation')
wave.rectPhaseGrating(P, np.pi/2)
wave.propagate(L2)

print('Plotting')

x = wave.X
I = np.abs(wave.Psi)**2

#plt.plot(x.T, I.T)
#plt.show()
#exit()

datamin = -30e-3
datamax = 30e-3
numbins = int(1000)
mybins = np.linspace(datamin, datamax, numbins)
myhist = np.zeros(numbins-1, dtype=np.double)
for xx, ii in zip(x, I):
    htemp, _ = np.histogram(xx, mybins, weights=ii)
    myhist += htemp
center = (mybins[:-1] + mybins[1:])/2
width = mybins[1]-mybins[0]
plt.bar(center*1e3, myhist, align='center', width=width*1e3)

C, fit = fitSin(center, myhist, P*L/D)
plt.plot(center*1e3, fit, '--', color='orange')

plt.xlabel('x [mm]')
plt.show()

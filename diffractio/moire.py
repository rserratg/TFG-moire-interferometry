# Example: 3-phase grating system

from diffractio import degrees, mm, plt, np, um
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_masks_X import Scalar_mask_X

# Generate plane wave
x0 = np.linspace(-5*mm, 5*mm, 1024*5)
wvl = 1 * um

u0 = Scalar_source_X(x=x0, wavelength=wvl)
u0.plane_wave(theta=0, z0=0)
#u0.draw(kind='field')

# Rectangualr window
rw = Scalar_mask_X(x0, wvl)
rw.slit(x0=0, size=2*mm)
#rw.draw(kind='field')

# Grating pi/2
gr1 = Scalar_mask_X(x0, wvl)
gr1.binary_grating(
    period = 1 * mm,
    amin = 1,
    amax = 1,
    phase = np.pi/2,
    x0 = 0,
    fill_factor = 0.5)
    
# Grating pi
gr2 = Scalar_mask_X(x0, wvl)
gr2.binary_grating(
    period = 1 * mm,
    amin = 1,
    amax = 1,
    phase = np.pi,
    x0 = 0,
    fill_factor = 0.5)

# Rectangular window
u1 = u0 * rw

# Initial propagation
u1.RS(z=500*mm, new_field=False)
u1.draw(kind='field')

# First grating
u2 = u1 * gr1

# Second propagation
u2.RS(z=1000*mm, new_field=False)
u2.draw(kind="field")

# Second grating
u3 = u2 * gr2

# Third propagation
u3.RS(z=1000*mm, new_field=False)
u3.draw(kind="field")

# Third grating
u4 = u3 * gr1

#Final propagation
u4.RS(z=500*mm, new_field=False)
u4.draw("field")

u4.draw("intensity")

# Draw
plt.show()


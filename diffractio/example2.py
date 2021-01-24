# Example 2: diffraction by a rectangular amplitude grating

from diffractio import degrees, mm, plt, np, um
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_masks_X import Scalar_mask_X

# Generate plane wave
x0 = np.linspace(-350*um, 350*um, 2048)
wvl = .6238 * um

u0 = Scalar_source_X(x=x0, wavelength=wvl)
u0.plane_wave(theta=0, z0=0)
#u0.draw(kind='field')

# Rectangualr window
#rw = Scalar_mask_X(x0, wvl)
#rw.slit(x0=0, size=600*um)
#rw.draw(kind='field')

# Apply grating
gr = Scalar_mask_X(x0, wvl)
gr.binary_grating(
    period = 40 * um,
    amin = 0,
    amax = 1,
    phase = 0,
    x0 = 0,
    fill_factor = 0.5)
#gr.draw(kind='field')

u1 = u0 * gr

# Propagate (near-field)

u2 = u1.RS(z=10*um, new_field=True)
u2.draw(kind='intensity', normalize=True)

# Far-field
#u3 = u1.fft(z=100*mm, remove0=False, new_field=True, shift=True)
#u3.draw(kind='amplitude',logarithm=False, normalize=True)

# Draw
plt.show()


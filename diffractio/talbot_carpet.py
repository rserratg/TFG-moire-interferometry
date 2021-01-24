from diffractio import np, plt, mm, um
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_fields_XZ import Scalar_field_XZ

x = np.linspace(-350*um, 350*um, 2048)
z = np.linspace(0*um, 10*mm, 512)
wvl = 0.6238 * um
period = 40 * um
z_talbot = 2*period**2/wvl

u0 = Scalar_source_X(x,wvl)
u0.plane_wave(A=1)

t = Scalar_mask_X(x,wvl)
t.ronchi_grating(period=40*um, x0=0*um, fill_factor=0.5)

talbot_effect = Scalar_field_XZ(x,z,wvl)
talbot_effect.incident_field(u0*t)
talbot_effect.BPM()

talbot_effect.draw(kind='intensity')
plt.ylim(-150*um,150*um)
plt.show()



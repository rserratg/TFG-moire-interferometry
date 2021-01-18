# Double slit

from diffractio import degrees, mm, plt, np, um
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.utils_drawing import draw_several_fields

from matplotlib import rcParams
rcParams['figure.dpi']=75

x0 = np.linspace(-2*mm, 2*mm, 1024*2)
wavelength = 1 * um

# plane wave
u0 = Scalar_source_X(x=x0, wavelength=wavelength)
u0.plane_wave(A=1, theta=0)

# slit

t0 = Scalar_mask_X(x=x0,  wavelength=wavelength)
t0.double_slit(x0=0, size=0.5 * mm, separation=1*mm)
t0.draw()

u1 = u0 * t0

u4=u1.RS(z=100*mm, new_field=True);
u4.draw(kind='field')

plt.show()

## optwave2D

Submodule for diffraction calculation with two-dimensional optical waves.

### Classes

- OptWave2D: main class.
- Mixins: auxiliary classes that contain functions from the main class. Allow to separate the class in multiple files. These classes should not be instantiated.

### Usage

1. Initialize wave object with OptWave2D constructor
2. Initialize wave function (wavefunctions)
3. Add optical elements (elements) and/or propagate (propagation)

Some simple cases can be compared with their analytical solutions (analytic results).
These functions are sampled in the current wave x-space.

### Class attributes
- dx: x-space sampling interval
- dy: y-space sampling interval
- x: x-space sampling points
- y: y-space sampling points
- wvl: wavelength
- U: wave function (complex) amplitude at sampling points

### Function list

- Main class:
    - OptWave2D: class constructor
- Wavefunctions:
    - planeWave: initialize wave as a plane wave
    - gaussianBeam: initialize wave as a Gaussian beam
- Elements:
    - rectSlit: single slit (rectangular aperture)
    - rectAmplitudeGratingX: binary amplitude grating, periodicity in x direction.
    - rectPhaseGratingX: binary phase grating, periodicity in x direction
    - lens: transparent lens, using thin lens and paraxial approximation
- Propagation:
    - fraunhofer: far-field Fraunhofer diffraction
    - fresnel_DI: Fresnel direct integration method
    - fresnel_CV: Fresnel convolution method
    - fresnel_AS: Fresnel convolution in frequency space
    - rayleigh_sommerfeld: Rayleigh-Sommerfeld formula
    - angular_spectrum: Angular Spectrum method
- Analytic solutions:
    - planeRectFresnelSolution: Fresnel diffraction of a plane wave from a single slit (rectangular aperture)

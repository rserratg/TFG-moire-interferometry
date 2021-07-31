## optwave1D

Submodule for diffraction calculation with one-dimensional optical waves.

### Classes

- OptWave: main class.
- Mixins: auxiliary classes that contain functions from the main class. Allow to separate the class in multiple files. These classes should not be instantiated.

### Usage

1. Initialize wave object with OptWave constructor
2. Initialize wave function (wavefunctions)
3. Add optical elements (elements) and/or propagate (propagation)

Some simple cases can be compared with their analytical solutions (analytic results).
These functions are sampled in the current wave x-space.

### Class attributes
- d: x-space sampling interval
- x: x-space sampling points
- wvl: wavelength
- U: wave function (complex) amplitude at sampling points

### Function list

- Main class:
    - OptWave: class constructor
- Wavefunctions:
    - planeWave: initialize wave as a plane wave
    - gaussianBeam: initialize wave as a Gaussian beam
- Elements:
    - rectAperture: single slit
    - sinAmplitudeGrating: sinusoidal amplitude grating
    - doubleSlit: double slit
    - rectAmplitudeGrating: binary amplitude grating
    - rectPhaseGrating: binary phase grating
    - lens: transparent lens, using thin lens and paraxial approximations
    - trapezoidPhaseObject: trapezoidal phase object
- Propagation:
    - fraunhofer: far-field Fraunhofer diffraction
    - fresnel_DI: Fresnel direct integration method
    - fresnel_CV: Fresnel convolution method
    - fresnel_AS: Fresnel convolution in frequency space
    - rayleigh_sommerfeld: Rayleigh-Sommerfeld formula
    - angular_spectrum: Angular Spectrum method
- Analytic solutions:
    - planeRectApertureSolution: Fraunhofer diffraction of a plane wave from a single slit
    - planeRectFresnelSolution: Fresnel diffraction of a plane wave from a single slit
    - planeSinAmpGrAnalyticSolution: Fraunhofer diffraction of a plane wave from a finite sinusoidal amplitude grating
    - planeDoubleSlitAnalyticSolution: Fraunhofer diffraction of a plane wave from a double slit
    - planeRectAmpGrAnalyticSolution: Fraunhofer diffraction of a plane wave from a finite binary amplitude grating

## neutronpckg

Package for neutron matter wave diffraction calculation (1D).

### Classes

- NeutronWave: main class.
- Mixins: auxiliary classes that contain functions from the main class. Allow to separate the class in multiple files. These classes should not be instantiated.

### Usage

1. Initialize wave object with NeutronWave constructor
2. Initialize wave function (wavefunctions)
3. Add optical elements (elements) and/or propagate (propagation)
4. Get intensity at camera (plotting)

### Class attributes
- Sampling space:
    - Nx: number of sampling points
    - d: x-space sampling interval
    - x: x-space sampling points (local space)
- Neutron properties:
    - Nn: number of neutrons
    - wvl: wavelength
    - theta: angle of propagation of each neutron w.r.t. z axis
    - x0: center of each neutron's space
- Experiment:
    - L: total propagation distance (from source)
    - Psi: wave function (complex) amplitude at sampling points
    - X: local sampling space of each neutron in global space coordinates

### Function list

- Main class:
    - NeutronWave: class constructor
- Wavefunctions:
    - slitSource: wavefunction at a distance from a single slit, in the Gaussian approximation.
- Elements:
    - slit: single slit
    - doubleSlit: double slit
    - rectPhseGrating: binary phase grating
- Propagation: 
    - propagate: propagation based on Fresnel approximation, convolution in frequency space
    - propagate_linear_potential: propagation inside a transverse linear potential, in the paraxial approximation
- Plotting:
    - hist_intensity: histogram of intensity (sum of all squared neutron wavefunctions)

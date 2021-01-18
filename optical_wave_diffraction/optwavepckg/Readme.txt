*********************************
*                               *
*  optwavepckg - OptWave class  *
*                               *
*********************************

 _________
|         |
|  Usage  |
|_________|

    1. Initialize wave object with OptWave constructor
    2. Initialize wave function (wavefunctions)
    3. Add optical element (elements) or propagate (propagation)
    
    Some simple cases can be compared with their analytical solutions (analytic results).
    This functions are sampled in the current wave x-space.


 _________________
|                 |
|  Function list  |
|_________________|

Main Class: 
    OptWave(N, L, lamb)
    
Analytic results:
    planeRectAnalyticSolution(z, Lx) : Uout
    planeRectFresnelSolution(z, Lx) : Uout
    
Elements:
    rectAperture(D)
    
Propagation:
    fraunhofer(z)
    fresnel_integral_one_step(z)
    fresnel_ang_spec(z)
    
Wavefunctions:
    planeWave(A=1)
    


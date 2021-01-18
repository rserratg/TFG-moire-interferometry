from numpy import random, outer, ones, linspace, tan, fft, cos,\
                    zeros, exp, sin
import numpy as np
import numpy.fft as afft
from math import pi
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Physical Constants
h_bar=1.0545718E-34; # [J*s]
m=1.674929E-27; # neutron mass [kg]
Kb=1.38064852E-23; #Boltzman [m**2 kg s**-2 K**-1]

class Experiment:
    def __init__(self, N_neutrons=1000, slit=1E-3, theta=0.2, 
                 wavelength=4.4E-10):
        '''
        Initialize expeiment object.
        
        Experiment object contains all information about a neutron experiment
        and contains functions that simulate 'actions' on a neutron beam.
        Initial parameters inclused number of neutrons in simulation, slit
        size, and beam divergence angle, theta.
        
        Initialization also creates a 1-D space in which each neutron
        wave-packet is defined and operated on.
        
        Each neutron is represented by a row in the saved matricies.
        Different rows corrispond to different neutrons.

        Parameters
        ----------
        N_neutrons : int, optional
            Number of neutrons in experiment. The default is 1000.
        slit : float, optional
            Width of slit at 'neutron source' [m]. The default is 1E-3.
        theta : float, optional
            Divergence angle of neutron beam [degrees]. The default is 0.2.
        wavelength : float, optional
            Mean wavelength of neutrons in experiment [m]. The default is 
            4.4E-10.

        '''
        print('Initializing experiment')
        
        #Define wave-packet space parameters
        X_limit=9E-3                                #Min and Max of x-space
        dX=0.15E-6                                  #x-space pixel size  
        
        NX=int(2*X_limit/dX)                        #Number of x-space points
        X=linspace(-X_limit, X_limit, NX)
        self.dx=dX        
        self.C=900;                                 #Group velocity [m/s]
        
        #Neutron Generation
        self.N_neutron=N_neutrons;      #Number of neutrons in simulation
        self.slit=slit;                 #Slit size at neutron source
        self.L=0;                       #Distance propogated.
        
        
        #Define each neutrons initial position (uniformly spaced over slit)
        X0=np.round((linspace(-self.slit/2,
                                    self.slit/2, 
                                    self.N_neutron))/self.dx);

        #Generate wavelength of each neutron from Maxwell dist. (unused)
        # scale=(h_bar*2*np.pi)**2/2/m/(5E-10)**2/Kb
        # Lambda=np.sqrt((h_bar*2*np.pi)**2/ \
        #     stat.maxwell.rvs(scale=scale/np.sqrt(2), 
        #                       size=self.N_neutron)/2/m/Kb)
        
        #Define wavelength of each neutron
        Lambda=wavelength*np.ones(self.N_neutron)
        
        #Generate initial angle of each neutron from Gaussian dist (unused)
        # Theta0=pi/180*theta*random.randn(self.N_neutron);
        
        #Generate intial angle of each neutron unform dist
        Theta0=pi/180*theta*(2*random.rand(self.N_neutron)-1);
        
        #Create matricies for x, x0, v0 from vectors above
        self.x=outer(X, ones(self.N_neutron));
        self.x0=outer(ones(len(X)), X0*self.dx);
        self.lamb=outer(ones(len(X)), Lambda);
        self.theta0=outer(ones(len(X)), Theta0);
        self.c=self.C/cos(self.theta0);
        
        #Find the frequency values for x-space
        K=fft.fftfreq(len(X), d=self.dx)*2*pi
        self.k=outer(K, ones(self.N_neutron));
        self.dk=1/max(X)*2*pi;
        
        #Define the larger camera space where neutrons will be projected
        final_limit = 0.01                      #Min/max of camera space [m]
        self.finalSpace=np.arange(-final_limit, final_limit+self.dx, self.dx)
        self.firstIndex=(len(self.finalSpace)-len(X))//2-X0.astype(int);
        
        
    def initialPropagation(self, L1, coherence_length=None):
        '''
        Propogate neutrons from slit to first object.
        The transverse coherence length of the neutron is increased by the 
        appropriate factor

        Parameters
        ----------
        L1 : float
            Distance from slit to first object.
        coherence_length : float, optional
            Manually specify the transverse coherence length at the first
            object. The default is None, and coherence length is calculated.

        '''
        print('Propogating neutron to first object')
        
        #Set total propgation length as distance from slit to first object
        self.L=L1;
        
        if coherence_length is None:
            sigma=self.lamb*L1/self.slit   # Transverse coherence length at L1
        else:
            sigma=coherence_length     # Use custom coherence length 
        
        # Wavefunction before first object
        self.Psi=(1/2/pi/sigma**2)**(1/4)*exp(-self.x**2/
                  (4*sigma**2)).astype(np.complex128);
    
    
    def absorptionGrating(self, P=2.4E-6, G=None):
        '''
        Apply absorption grating to each neutron. Grating profile is shifted
        based on inital position, angle, and distance propagated.

        Parameters
        ----------
        P : float
            Period of square grating [m]. The default is 2.4E-6.
        G : ndarray, optional
            Manually specify grating vector, if None the grating is square. 
            Vector length must be equal to the length of x-space. The default 
            is None.

        '''
        print('Applying absorption grating')
        
        if G is None:
            G=(cos(2*pi/P*(self.x-self.x0-self.L*tan(self.theta0)))>=0);
        self.Psi=self.Psi*G;


    def phaseGrating(self, P=2.4E-6, N=4.99591E28, bc=4.107E-15, h=15E-6,
                     G=None, phi=None):
        '''
        Apply phase grating to each neutron. Grating profile is shifted
        based on inital position, angle, and distance propagated.

        Parameters
        ----------
        P : float, optional
            Period of square phase grating. The default is 2.4E-6.
        N : float, optional
            Number density of material in phase grating []. The default is 
            4.99591E28 (Silicon).
        bc : float, optional
            Scattering length density of material in phase grating []. The 
            default is 4.107E-15.
        h : float, optional
            Height of phase grating [m]. The default is 15E-6.
        G : ndarray, optional
            Manually specify grating vector, if None the grating is square. 
            Vector length must be equal to the length of x-space. The default 
            is None.
        phi : float, optional
            Manually specify the phase shift caused by grating, if None the 
            strength is calculated. The default is None.


        '''
        print('Applying phase grating')
        if G is None:
            G=(cos(2*pi/P*(self.x-self.x0-self.L*tan(self.theta0)))>=0);
        if phi is None:
            self.Psi=self.Psi*exp(-1j*(N*bc*self.lamb*h)*G);
        else:
            self.Psi=self.Psi*exp(-1j*phi*G);
    
    
    def magneticPrism(self, B=100E-4, alpha=60):
        '''
        

        Parameters
        ----------
        B : float, optional
            Strength of magnetic fiel in prism [T]. The default is 100E-4.
        alpha : float, optional
            Angle of prism (relative to Y-axis) [degrees]. The default is 60.

        '''
        print('Applying magnetic coil')
        
        pauli_X=np.array([[0,1],[1,0]]);    #Define spin rotation matrix (X)
        alpha=alpha*180/pi                  #Convert alpha to radians
        beta=pi/2-alpha                     #Calculate second prism angle
        
        #Create vector to store phase shifts for each neutron
        F=ones(self.N_neutron, dtype=np.complex128)
        
        for i in range(self.N_neutron):
            #Calculate path-length in prism for each neutron
            L=sin(alpha)/sin(beta-self.theta0[0,i])*(1E-2 
                                                 + self.L*tan(self.theta0[0,i])
                                                 + self.x0[0,i])

            #Calculate resulting spin-vector for neutron (initally [1,0])
            prism=expm(1j*1.832E8*B*m*self.lamb[0,i]*L/h_bar/
                 pi/4*pauli_X);
            
            #Polarize beam after prism (measure [1,0])
            F[i]=prism[1, 0]
            
        #Generate matrix of attenuation coefficients for each neutron
        self.f=outer(ones(len(self.x)), F)
        
        #Apply attenuation
        self.Psi=self.Psi*self.f
            

    def propagate(self, d):
        '''
        

        Parameters
        ----------
        d : float
            Distance of neutron propagation.

        '''
        print('Time propagating neutron')
        
        #Wavefunction in k-space
        PsiK=afft.ifftshift(afft.fft(afft.fftshift(self.Psi), axis=0))
        
        #Propagation kernel
        U_t=exp(-1j*h_bar*self.k**2*d/(self.C/cos(self.theta0))/2/m);
        
        #Add to travelled distance
        self.L+=d;
        
        #Calculate new wavefunction of each neutron
        self.Psi=afft.fftshift(afft.ifft(afft.ifftshift(PsiK)*U_t, axis=0))


    def plotWave(self,  title='Wavefunction Intensities'):
        '''
        Plot the intenities of all neutron wavefunctions. Each wavefunction
        is embedded in the camera space based on its shift during progagation.

        Parameters
        ----------
        title : string, optional
            Title of plot. The default is 'Wavefunction Intensities'.

        '''
        print('Measuring')
        
        #Create space to combine neutron wavefunctions
        PsiF=zeros((len(self.finalSpace)), dtype=np.float64);
        
        #Calcuate shift of each neutron over total propagation distance
        XShift=self.L*tan(self.theta0[0,:]);
        
        #Index to begin embedding wavefunctions
        Index=self.firstIndex-np.round(XShift/self.dx).astype(int);
        
        #Embded each neutron in the camera space
        for j in range(self.N_neutron):
            begin=0;                            #First index in x-space
            end=len(self.x);                    #Last index in x-space

            #Ignore points outside of camera space
            if Index[j]+len(self.x)<0:
                continue;
            elif Index[j]<0:
                begin=-Index[j];
                
            if Index[j]>len(self.finalSpace):
                continue;
            elif Index[j]+len(self.x)>len(self.finalSpace):
                end=len(self.x)-(Index[j]+len(self.x)-
                        len(self.finalSpace));
            
            #Add wavefunction intensity to camera space 
            PsiF[(Index[j]+begin):(Index[j]+end)]+=np.abs(
                self.Psi[begin:end,j])**2;
        self.Psisq=PsiF;                                #Store intensities
        
        print('Ploting wavefunctions')
        plt.figure();
        plt.plot(self.finalSpace, self.Psisq)
        plt.xlabel('X-Distance [m]')
        plt.ylabel('Intensity')
        plt.title(title)

        plt.show()

    def cameraHist(self, title='Counts', ):
        '''
        Sample from each neutron wavefuntion to generate 'counts' on camera
        space. Each neutron adds one count.

        Parameters
        ----------
        title : string, optional
            Title of plot. The default is 'Counts'.

        '''
        print('Binning wavefunction and plotting image')
        
        #Measured positions of each neutron
        pos=zeros(self.N_neutron);
        
        #Calcuate shift of each neutron over total propagation distance
        XShift=self.L*tan(self.theta0[0,:]);    
        
        #Index to begin embedding wavefunctions
        Index=self.firstIndex-np.round(XShift/self.dx).astype(int);
        
        for j in range(self.N_neutron):
            #Position probabilites
            Psi=abs(self.Psi[:,j])**2*self.dx/ \
                sum(abs(self.Psi[:,j])**2*self.dx);
                
            #Measured position
            pos[j]=random.choice(self.x[:,0], replace=False, p=Psi)\
                +(Index[j]*self.dx+min(self.finalSpace));
        
        plt.figure(2)
        plt.hist(pos, bins=int(33) )
        plt.xlabel('Position [m]')
        plt.ylabel('Count')
        plt.title(title)


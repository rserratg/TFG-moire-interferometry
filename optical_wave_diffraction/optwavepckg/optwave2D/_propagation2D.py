# Propagation methods for the OptWave2D class

import numpy as np

'''
    Helper (Mixin) class for the OptWave2D class

    Contains the methods related to wave propagation methods
'''

class MixinProp2D:

    # AUXILIARY METHODS: CONVOLUTION CALCULATIONS

    '''
        Propagate field using impulse response
        (Convolution fft calculation for propagation methods)
        Using zero-padding and, optionally, Simpson's rule.

        S = IFFT[FFT(U) * FFT(H)]

        Parameters:
            - kernel (function(x,y)): impulse response.
                Parameter x is x-space.
                Parameter y is y-space.
            - simpson (bool):
                if True, uses Simpson's rule for better accuracy
                By default False.

        Note:
            - When using Simpson, it is highly advised to use Nx, Ny odd

        References:
            - F. Shen & A. Wang. https://doi.org/10.1364/AO.45.001102
    '''
    def _fft_conv(self, kernel, simpson=False):
        
        Nx = len(self.x)
        Ny = len(self.y)
        x = self.x
        y = self.y
        u = self.U

        # Simpson's integration rule
        if simpson:
            a = [2,4]

            num_repx = int(round(Nx/2)-1)
            bx = np.array(a * num_repx)
            Bx = np.concatenate(((1,), bx, (2,1))) / 3.
            if float(Nx) / 2 == round(Nx/2): # is even
                i_central = num_repx + 1
                Bx = np.concatenate((Bx[:i_central], Bx[i_central+1:]))

            num_repy = int(round(Ny/2)-1)
            by = np.array(a * num_repy)
            By = np.concatenate(((1,), by, (2,1))) / 3.
            if float(Ny) / 2 == round(Ny/2): # is even
                i_central = num_repy + 1
                By = np.concatenate((By[:i_central], By[i_central+1:]))        

            W = np.outer(By, Bx)
            u = W*u

        # Zero-padding

        # Field
        U = np.zeros((2*Ny-1, 2*Nx-1), dtype=np.complex128)
        U[0:Ny, 0:Nx] = u

        # Double domain
        xext = self.x[0] - x[::-1]
        xext = xext[0:-1]
        x = np.concatenate((xext, self.x - x[0]))

        yext = self.y[0] - y[::-1]
        yext = yext[0:-1]
        y = np.concatenate((yext, self.y - y[0]))

        # H = kernel in freq-space
        H = np.fft.fft2(kernel(x,y))

        S = np.fft.ifft2(np.fft.fft2(U) * H) * self.dx * self.dy
        self.U = S[Ny-1:, Nx-1:]


    '''
        Propagate field using transfer function.
        (Convolution fft calculation for propagation methods).
        Using zero-padding and, optionally, Simpson's rule

        S = IFFT[FFT[U]*H]

        Parameters:
            - kernel (function(fx,fy)): transfer function.
                Parameter fx is freq-space in x
                Parameter fy is freq-space in y
            - bandlimit (False or 2-double tuple):
                if double, delete components above given limits (fxmax,fymax)
                if False, don't delete any frequency component
                By default False.
            - simpson (bool):
                if True, uses Simpson's rule for better accuracy
                By default False.

        Note:
            - If fxmax or fymax = 0, all frequency components are eliminated in that direction (except the 0 freq.).
            - Important: notice that the domain is doubled due to zero-padding.
              Therefore, df'=2*df
            - When using Simpson rule, it is highly advises to use Nx and Ny odd
    '''
    def _ifft_conv(self, kernel, bandlimit=False, simpson=False):
        
        Nx = len(self.x)
        Ny = len(self.y)
        u = self.U

        # Simpson's integration rule
        if simpson:
            a = [2,4]

            num_repx = int(round(Nx/2)-1)
            bx = np.array(a * num_repx)
            Bx = np.concatenate(((1,), bx, (2,1))) / 3.
            if float(Nx) / 2 == round(Nx/2): # is even
                i_central = num_repx + 1
                Bx = np.concatenate((Bx[:i_central], Bx[i_central+1:]))

            num_repy = int(round(Ny/2)-1)
            by = np.array(a * num_repy)
            By = np.concatenate(((1,), by, (2,1))) / 3.
            if float(Ny) / 2 == round(Ny/2): # is even
                i_central = num_repy + 1
                By = np.concatenate((By[:i_central], By[i_central+1:]))        

            W = np.outer(By, Bx)
            u = W*u

        # Zero-padding

        # Field
        U = np.zeros((2*Ny-1, 2*Nx-1), dtype=np.complex128)
        U[0:Ny, 0:Nx] = u 

        # Double frequency domain
        # Freq-space equivalent to doubling x-space domain
        fx = np.fft.fftfreq(2*Nx-1, self.dx)
        fy = np.fft.fftfreq(2*Ny-1, self.dy)

        # H = kernel in freq-space
        H = kernel(fx, fy)

        if bandlimit:
            fxmax, fymax = bandlimit
            cond1 = np.abs(fx) > fxmax
            cond2 = np.abs(fy) > fymax
            C1, C2 = np.meshgrid(cond1, cond2)
            H = np.where(C1 | C2, 0, H)

        S = np.fft.ifft2(np.fft.fft2(U)*H)
        self.U = S[0:Ny, 0:Nx]
        

    '''
        Propagate field using impulse response.
        (Convolution fft calculation for propagation methods).
        Not using zero padding nor Simpson's rule

        S = IFFT[FFT(U)*FFT(H)]

        Parameters:
            - kernel (function(x, y)): impulse response.
                Parameter x is x-space.
                Parameter y is y-space.

        Note:
            - Faster than using zero padding.
            - Might produce aliasing errors.
    '''
    def _fft_conv_nopad(self, kernel):

        x = self.x
        y = self.y
        u = self.U

        H = np.fft.fft2(kernel(x,y))
        U = np.fft.fft2(u)
        S = np.fft.ifft2(H*U)*self.dx*self.dy

        S = np.fft.fftshift(S)
        self.U = S

    
    '''
        Propagate field using transfer function.
        (Convolution fft calculation for propagation methods)
        Not using zero-padding nor Simpson's rule.

        S = IFFT[FFT(U)*H]

        Parameters:
            - kernel (function (fx,fy)): transfer function
                Parameter fx is frequency space in x.
                Parameter fy is frequency space in y.
            - bandlimit (False or 2-double tuple)
                if tuple, delete components above given limits (fxmax, fymax)
                if False, don't delete any frequency component
                By default False.

        Note:
            - Faster than using zero-padding
            - Might produce aliasing errors
    '''
    def _ifft_conv_nopad(self, kernel, bandlimit=False):
        
        Nx = len(self.x)
        Ny = len(self.y)
        u = self.U

        fx = np.fft.fftfreq(Nx, self.dx)
        fy = np.fft.fftfreq(Ny, self.dy)

        H = kernel(fx, fy)

        if bandlimit:
            fxmax, fymax = bandlimit
            cond1 = np.abs(fx) > fxmax
            cond2 = np.abs(fy) > fymax
            C1, C2 = np.meshgrid(cond1, cond2)
            H = np.where(C1 | C2, 0, H)

        S = np.fft.ifft2(np.fft.fft2(u)*H)
        self.U = S

    # ------------------------------------------------------------------------------------

    # AUXILIARY METHODS: PROPAGATION CONDITIONS

    '''
        Bandwidth limit for far-field propagation using Angular Spectrum method

        u_limit = 1/[sqrt((2*df*z)**2+1)*wvl]
        
        Parameters:
            - z (double): propagation distance
            - dfx (double): sampling interval in x-frequency-space
            - dfy (double): sampling interval in y-frequency-space

        References:
            - Matsushima 2009. Band-limited angular spectrum method for
              numerical simulation of free-space propagation in far and
              near fields.

        Notes:
            - Approximation valid for Lx, Ly << 2*z.
              See reference for details.
    '''
    def _bandlimitAS(self, z, dfx, dfy):
        ulim = (2*dfx*z)**2 + 1
        ulim = self.wvl * np.sqrt(ulim)

        vlim = (2*dfy*z)**2 + 1
        vlim = self.wvl * np.sqrt(vlim)

        return (1/ulim, 1/vlim)

    # --------------------------------------------------------------------------------

    # AUXILIARY METHODS: PROPAGATION KERNELS

    '''
        Propagation kernel: Fresnel convolution

        Real space.
        Paraxial approximation.

        Parameters:
            - z (double): propagation distance
            - x (np.array): x-space
    '''
    def _kernelFresnelCV(self, z, x, y):
        k = 2*np.pi/self.wvl 
        
        x2 = x**2
        y2 = y**2
        X2, Y2 = np.meshgrid(x2,y2)
        rho2 = X2 + Y2

        H = np.exp(1j*k/(2*z)*rho2)
        A = np.exp(1j*k*z)/(1j*self.wvl*z)

        return A*H


    '''
        Propagation kernel: Fresnel AS

        Frequency space.
        Paraxial approximation.

        Parameters:
            - z (double): propagation distance
            - fx (np.array): x freq space
            - fy (np.array): y freq space
    '''
    def _kernelFresnelAS(self, z, fx, fy):
        k = 2 * np.pi / self.wvl 

        fx2 = fx**2
        fy2 = fy**2
        Fx2, Fy2 = np.meshgrid(fx2, fy2)
        Fsq = Fx2 + Fy2

        H = np.exp(-1j * np.pi * self.wvl * z * Fsq)
        A = np.exp(1j*k*z)

        return A*H

    '''
        Propagation kernel: Rayleigh-Sommerfeld

        Real space.
        Assuming R >> wvl.

        Parameters:
            - z (double): propagation distance
            - x (np.array): x-space
            - y (np.array): y-space
    '''
    def _kernelRS(self, z, x, y):
        k = 2 * np.pi / self.wvl 

        x2 = x**2
        y2 = y**2
        X2, Y2 = np.meshgrid(x2, y2)
        R2 = X2 + Y2 + z**2
        R = np.sqrt(R2)

        H = z*np.exp(1j*k*R)/R2
        A = 1/(1j*self.wvl)
        
        return A*H

    
    '''
        Propagation kernel: Angular Spectrum representation

        Frequency space.

        Parameters:
            - z (double): propagation distance
            - fx (np.array): x space freq
            - fy (np.array): y space freq
    '''
    def _kernelAS(self, z, fx, fy):
        fx2 = fx**2
        fy2 = fy**2
        Fx2, Fy2 = np.meshgrid(fx2, fy2)
        Fsq = Fx2 + Fy2
        
        Fsq = np.where(Fsq>1/self.wvl**2, 0, Fsq) # remove evanescent waves
        m = np.sqrt(1/self.wvl**2 - Fsq)

        H = np.exp(1j*2*np.pi*z*m)
        return H 


    # ------------------------------------------------------------------------------------------

    # AUXILIARY METHODS: GENERAL PROPAGATION FUNCTION

    '''
        General function for propagation using comvolution method

        Parameters:
            - method (string): propagation method - "AS", "RS", "FresnelAS", "FresnelCV"
            - z (double): propagation distance
            - pad (bool): if True, use zero-padding. By default True.
            - Conditional parameters:
                - bandlimit (bool or 2-double tuple):
                    Only if method="AS" or method="FresnelAS" (kernel in freq space)
                    if False, use all frequency components.
                    if tuple, use only frequency components up to given limits (fxmax, fymax)
                    if True, calculate max frequencies to be used.
                    By default True.
                - simpson: 
                    Only if pad=True (zero-padding is used).
                    if True, use Simpson rule for improved accuracy
                    (note that it doesn't guarantee better results)
                    By default False.
    '''
    def _conv_propagation(self, z, method, pad=True, **kwargs):
        
        # If True, kernel in real space (RS, FresnelCV)
        # If False, kernel in frequency domain (AS, FresnelAS)
        kernelspace = True

        # Kernel function
        kernel = None
        if method == "AS":
            kernel = lambda fx, fy : self._kernelAS(z, fx, fy)
            kernelspace = False
        elif method == "RS":
            kernel = lambda x, y : self._kernelRS(z, x, y)
        elif method == "FresnelAS":
            kernel = lambda fx, fy : self._kernelFresnelAS(z, fx, fy)
            kernelspace = False
        elif method == "FresnelCV":
            kernel = lambda x, y : self._kernelFresnelCV(z, x, y)
        else:
            print("Propagation test: invalid kernel")
            return 

        # Propagation
        if kernelspace:
            if pad:
                simpson = kwargs.get("simpson", False)
                self._fft_conv(kernel, simpson)
            else:
                self._fft_conv_nopad(kernel)
        else:
            # If bandlimit = True: calculate max freq
            bandlimit = kwargs.get("bandlimit", True)
            if (type(bandlimit) is bool) and bandlimit:
                Nx = len(self.x)
                Ny = len(self.y)
                dfx = 1/(self.dx*Nx)
                dfy = 1/(self.dy*Ny)
                if pad: # if zero-padding is used, domain is doubled
                    dfx /= 2
                    dfy /= 2
                bandlimit = self._bandlimitAS(z, dfx, dfy)
            
            if pad:
                simpson = kwargs.get("simpson", False)
                self._ifft_conv(kernel, bandlimit, simpson)
            else:
                self._ifft_conv_nopad(kernel, bandlimit)


    # ---------------------------------------------------------------------------

    # PROPAGATION METHODS

    '''
        Propagation of optical waves based on the far-field Fraunhofer approximation
        (2D)

        Parameters:
            - z (double): propagation distance

        Post:
            - self.U = wave function after propagation
            - self.x, self.y, self.dx, self.dy updated

        Notes:
            - Region of validity: z > 2D*2/lambda 
              (lambda = wavelen, D = aperture diameter)
            - Valid near origin (x,y) ~ (0,0)
    '''
    def fraunhofer(self, z):

        Nx = len(self.x)
        Ny = len(self.y)
        k = 2*np.pi/self.wvl 

        fx = np.fft.fftshift(np.fft.fftfreq(Nx, self.dx))
        fy = np.fft.fftshift(np.fft.fftfreq(Ny, self.dy))

        # Observation plane coordinates
        self.x = self.wvl * z * fx
        self.y = self.wvl * z * fy

        x2 = self.x**2
        y2 = self.y**2
        X2, Y2 = np.meshgrid(x2, y2)
        Rho2 = X2 + Y2

        A = np.exp(1j*k*z)/(1j*self.wvl*z)
        FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.U))) * self.dx * self.dy
        
        self.U = A*np.exp(1j*k/(2*z)*Rho2)*FT
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]


    '''
        Propagation of optical waves based on the Fresnel approximation.
        Direct integration via fft, one-step integral method.
        (2D)

        Parameters:
            - z (double): propagation distance

        Post:
            - self.U = wave function after propagation
            - self.d = self.wvl*z/(N*self.d)
            - self.x / self.y updated with new d

        Notes:
            - Suited only for far-field amplitude evaluation.
    '''
    def fresnel_DI(self, z):

        Nx = len(self.x)
        Ny = len(self.y)
        k = 2*np.pi/self.wvl

        # Source plane coordinates
        xin = self.x 
        yin = self.y

        xin2 = xin**2
        yin2 = yin**2
        Xin2, Yin2 = np.meshgrid(xin2, yin2)
        Rin2 = Xin2 + Yin2

        # Observation plane coordinates
        xout = self.wvl*z*np.fft.fftshift(np.fft.fftfreq(Nx, self.dx))
        yout = self.wvl*z*np.fft.fftshift(np.fft.fftfreq(Ny, self.dy))

        xout2 = xout**2
        yout2 = yout**2
        Xout2, Yout2 = np.meshgrid(xout2, yout2)
        Rout2 = Xout2 + Yout2

        A = np.exp(1j*k*z)*np.exp(1j*k/(2*z)*Rout2)/(1j*self.wvl*z)

        u = self.U * np.exp(1j*k/(2*z)*Rin2)
        FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u))) * self.dx * self.dy 
        self.U = A*FT

        self.x = xout
        self.y = yout

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]


    '''
        Propagation of optical waves based on the Fresnel approximation
        Direct integration method via convolution
        (2D)

        Parameters:
            - z (double): propagation distance
            - simpson (bool): if True, use Simpson rule. By default False.

        Post:
            - self.U = wave function after propagation

        Notes:
            - dout = din, xout = xin.
            - Sampling condition: d <= wvl*z/L
            - Zero padding is used
    '''
    def fresnel_CV(self, z, simpson=False):
        self._conv_propagation(z, "FresnelCV", simpson=simpson)


    '''
        Propagation of optical waves based on the Fresnel approximation
        Angular Spectrum method (convolution theorem)
        (2D)

        Parameters:
            - z (double): propagation distance
            - bandlimit (bool or 2-double tuple):
                if tuple, delete frequency components above given limits (fxmax, fymax)
                if True, calculate frequency limit and delete higher-frequency components
                if False, don't delete any frequencies
                By default True.
            - simpson (bool): if True, use simpson rule. By default False.

        Post:
            - self.U = wave function after propagation

        Notes:
            - dout = din, xin
            - Sampling condition: d >= sqrt(wvl*z/N)
            - Zero padding is used
    '''
    def fresnel_AS(self, z, bandlimit=True, simpson=False):
        self._conv_propagation(z, "FresnelAS", bandlimit=bandlimit, simpson=simpson)


    '''
        Propagation of optical waves based on Rayleigh-Sommerfeld I formula. 
        FFT convolution method.

        Parameters:
            - z (double): propagation distance
            - simpson (bool): if True, use Simpson rule. By default False.

        Post:
            - self.U = wave function after propagation

        Notes:
            - Assuming n=1.
            - dout = din, xout = xin
            - Sampling condition: d <= wvl*sqrt(z**2+L**2/2)/L
            - Zero padding is used
    '''
    def rayleigh_sommerfeld(self, z, simpson=False):
        self._conv_propagation(z, "RS", simpson=simpson)

    
    '''
        Propagation of optical waves using angular spectrum representation.
        Convolution method with analytical transfer function.
        (2D)

        H = exp(1j*pi*z*m) where m = sqrt(1/wvl^2 - fx^2 - fy^2)

        Parameters:
            - z (double): propagation distance
            - bandlimit (bool or 2-double tuple):
                if tuple, delete frequency components above given limits (fxmax, fymax)
                if True, calculate frequency limit and delete higher-frequency components
                if False, don't delete any frequency component
                By default True.
            - simpson (bool): if True, use Simpson rule. By default False.

        Post:
            - self.U = wave function after propagation

        Notes:
            - dout = din, xout = xin
            - Sampling condition: wvl*z*N/(L*sqrt(L**2-2*(wvl*N/2)**2)) <= 1
            - Bandwidth condition: f_limit = 1/(wvl*sqrt(1+(2*df*z)**2))
              Assuming L << 2*z (see reference)
              Bandwidth condition can be applied for good results in far field.
            - Zero padding is used. Notice that, in this case, df' = df/2

        References:
            - K Matsushima, T Shimobaba. Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields.
    '''
    def angular_spectrum(self, z, bandlimit=True, simpson=False):
        self._conv_propagation(z, "AS", bandlimit=bandlimit, simpson=simpson)

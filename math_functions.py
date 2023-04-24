import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fft2, ifft2, fftn, ifftn
from scipy.fftpack import fftshift, ifftshift
from mpl_toolkits.mplot3d import Axes3D
# from mayavi import mlab
import scipy.optimize as opt
from PIL import Image

def sampleVideo (matrix, frames, pause = .05):
    im=plt.imshow(np.abs(matrix[:,:,0])/np.amax(np.abs(matrix)), \
                        cmap = 'gray', interpolation = 'none')
    for i in range (frames):
        im.set_data(np.abs(matrix[:,:,i])/np.amax(np.abs(matrix)))
        plt.pause(0.5)
        #print i
    plt.show()
    return None
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
from inspect import currentframe
def debug_hint():
    frameinfo = currentframe()
    print ('\n{line: ', frameinfo.f_back.f_lineno,'}\n')
    return None
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def FT(f, ax = 0):
    return fftshift(fft(ifftshift(f), axis = ax))

def FT2(f):
    return fftshift(fft2(ifftshift(f)))

def FT3(f):
    return fftshift(fftn(ifftshift(f)))

def IFT(F, ax = 0):
    return ifftshift(ifft(fftshift(F), axis = ax))

def IFT2(F):
    return ifftshift(ifft2(fftshift(F)))

def IFT3(F):
    return ifftshift(ifftn(fftshift(F)))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def gaus(x, p = [0,1]):
    """Gaussian with:
                      - x = domain;
                      - p[0] = center;
                      - p[1] = width.
    """

    return np.exp(-np.pi*(x-p[0])**2/(p[1]**2))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def gaus_2d(X, Y, p=[1, 0,  0., .5, .5]):
    return p[0] * \
            np.exp(-(np.pi*(X-p[1])**2)/(np.abs(p[3])**2)-\
                                    (np.pi*(Y-p[2])**2)/(np.abs(p[4])**2))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def gaus_2d_forFit(X, Y, p0, p1, p2, p3, p4, p5):
    return (p0 * np.exp(-(np.pi*(X-p1)**2)/(p3**2)-\
                          (np.pi*(Y-p2)**2)/(p4**2))+p5).ravel()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def double_gaus_2d_forFit(X, Y, p=[0, 1, 2, 3, 4, 5, 6, 7, 1, 1]):
    return (p[8]*np.exp(-(np.pi*(X-p[0])**2)/\
                                    (p[1]**2)-(np.pi*(Y-p[2])**2)/(p[3]**2))\
            +p[9]*np.exp(-(np.pi*(X-p[4])**2)/\
                                    (p[5]**2)-(np.pi*(Y-p[6])**2)/(p[7]**2)))\
            .ravel()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def step(x):
    """Step function:
                    - x = domain.
    """
    return .5*(1+sp.sign(x))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def rect(x, p=[0.,1.]):
    """Rectangular function
                      - x = domain;
                      - p[0] = center;
                      - p[1] = width.
    """
    return step((x-p[0])/p[1]+.5)-step((x-p[0])/p[1]-.5)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def rect_2d(x, y, p=[0., 0., 1., 1.]):
    """Rectangular function
                      - x, y = 2d grid domain;
                      - p[0] = center_x;
                      - p[1] = center_y;
                      - p[2] = width_x;
                      - p[3] = width_y;
    """

    return (step((x[:]-p[0])/p[2]+.5)-step((x[:]-p[0])/p[2]-.5))*\
                        (step((y[:]-p[1])/p[3]+.5)-step((y[:]-p[1])/p[3]-.5))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def circ(x, y,center, D):
    """Circ function
                      - x, y = 2d grid domain;
                      - D = mask diamenter.
    """
    out = np.zeros((x.shape[0], y.shape[1]))
    for i in range (out.shape[0]):
        for j in range (out.shape[1]):
            if((x[i,j]-center)**2+(y[i,j]-center)**2)< (D/2.)**2 :
                out[i,j] = 1.
    return out
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def crossCorr (f, g):
    """Cross-correlation function, with padding. For flows: it will
        highligh a flow of particle f --> g;
                        - f = first signal;
                        - g = second signal.
        """

    N = len(f)
    one, two = np.pad(np.copy(f),\
                    (N/2),\
                            mode = 'constant', constant_values=(0)),\
               np.pad(np.copy(g),\
                    (N/2),\
                            mode = 'constant', constant_values=(0))
    F, G = FT(one), FT(two)

    cross = np.real(ifft(ifftshift(F)*np.conj(ifftshift(G))))[:N]

    for i in range (len(cross)):
        cross[i] = cross[i]/(N-i)

    return cross
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def crossCorrFluct (f, g):
    """Cross-correlation function of the
        fluctuations of two signals, with padding, normalized.
        For flows: it will highligh a flow of particle f --> g;
                        - f = first signal;
                        - g = second signal.
        """

    N = len(f)
    mean_f, mean_g = np.mean(f), np.mean(g)
    one, two = np.pad(np.copy(f-mean_f),\
                    (N/2),\
                            mode = 'constant', constant_values=(0)),\
               np.pad(np.copy(g-mean_g),\
                    (N/2),\
                            mode = 'constant', constant_values=(0))
    F, G = fft(two), fft(one)

    cross = np.real(ifft(F*np.conj(G)))[:N]

    for i in range (len(cross)):
        cross[i] = cross[i]/(N-i)/mean_f/mean_g

    return cross
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def spatial_Xcorr_2(matrix_1, matrix_2):
    """
    To be tested
    """
    M, N = matrix_1.shape[0], matrix_1.shape[1]

    one, two = np.pad(np.copy(matrix_1),\
                    ((M/2, M/2),(N/2, N/2)),\
                            mode = 'constant', constant_values=(0,0)),\
               np.pad(np.copy(matrix_2),\
                    ((M/2, M/2),(N/2, N/2)),\
                            mode = 'constant', constant_values=(0,0))
    ONE, TWO =   FT2(one), FT2(two)

    spatial_cross = ifftshift(ifft2(ifftshift(ONE) * np.conj(ifftshift(TWO))))

    return spatial_cross[M/2 :M/2+matrix_1.shape[0],\
                        N/2 : N/2+matrix_1.shape[1]]
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def gausFit (vector, x_step = 1., plot = 'yes', title = 'titolo'):
    """Fitting with a gaussian:
                      - vector = what to fit;
                      - x_step (1.) = independent variable increment;
                      - plot ('yes') = show/hide a plot of the result;
                      - titel ('titolo') = name of the function.
                      - return = {[amplitude, center, sigma], iterations}
    """
    N = len(vector)
    x = np.arange(0., N, 1.)

    p0 = [0.,0.,0.] # A, x0, sigma
    p0[0] = np.amax(vector)
    p0[1] = np.argmax(vector)
    p0[2] = p0[0]/2.

    fit_func = lambda p,x: p[0]*np.exp(-np.pi*(x-p[1])**2/(p[2]**2))
    err_func = lambda p,x,y: fit_func(p,x)-y
    p1, success = sp.optimize.leastsq(err_func, p0, args=(x, vector))

    if (plot=='yes'):
        plt.figure('FITResults '+ title)
        plt.title('gausFit'+' '+title)
        plt.ylabel('Function'), plt.xlabel('x')

        plt.plot(x*x_step, vector,'ro',label = 'data')

        plt.plot(x*x_step,fit_func(p1, x),'b--',label = 'fit')

        plt.ylim (np.amin(vector)-.05, np.amax(vector)+.2)

        plt.annotate('Amplitude: '+str(p1[0])+'\n'\
                    +'Center: '+str(p1[1])+'\n'+
                    'Sigma:'+str(p1[2])+'\n',\
                    [p1[1],p1[0]], [1,p1[0]-.5])
        plt.grid(which = 'minor')
        plt.legend()
        plt.show()

    return p1, success
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def gausFit_2D (matrix, x, y, plot = 'yes', title = 'fit'):
    """Fitting with a gaussian:
                      - vector = what to fit;
                      - x_step (1.) = independent variable increment;
                      - plot ('yes') = show/hide a plot of the result;
                      - titel ('titolo') = name of the function.
                      - return = {[amplitude, center, sigma], iterations}
    """
    vector = matrix.ravel()

    # Initial guesses
    p0 = [1., 2. , 2., 4., 4., .5] # Amplitude, x0, y0, sigmaX, sigmaY
    p0[0] = np.amax(vector)
    p0[2], p0[1] = np.unravel_index(np.argmax(matrix), matrix.shape)
    p0[5] = np.amin(vector)

    p0 = tuple(p0)
    popt = 0

    popt, pcov = opt.curve_fit(gaus_2d_forFit, (x, y), vector, p0=p0)
    errs = np.sqrt(np.diag(pcov))

    if (plot=='yes'):
        after = gaus_2d((x, y), list(popt))
        plt.figure('FIT Results '+ title)
        plt.title(title)
        plt.ylabel('x'), plt.xlabel('y')

        plt.imshow(matrix, label = 'data', cmap = 'gray', \
            extent = (np.amin(y), np.amax(y), np.amin(x), np.amax(x)))
        levels = (np.linspace(np.amin(after), np.amax(after), 6 ))
        plt.contour(after[::-1,:], label = 'fit', cmap = 'hot', alpha = .5,\
                    levels = levels,\
                    extent = (np.amin(y), np.amax(y), np.amin(x), np.amax(x)))

        plt.annotate(\
                r'$A\,exp(-\frac{\pi(x-x_{0})^{2}}{\sigma_{x}^{2}}\,-\,$'+\
                r'$\frac{\pi(y-y_{0})^{2}}{\sigma_{y}^{2}})$'+'\n'+
                r'$A$= '+str(round(popt[0],3))+\
                r'$\,\pm\,$'+str(round(errs[0],4))+'\n'+\
                r'$x_{0}$= '+str(round(popt[1],4))+\
                r'$\,\pm\,$'+str(round(errs[1],4))+'\n'+\
                r'$y_{0}$= '+str(round(popt[2],4))+\
                r'$\,\pm\,$'+str(round(errs[2],4))+'\n'+\
                r'$\sigma_{x}= $'+str(round(popt[3],3))+\
                r'$\,\pm\,$'+str(round(errs[3],4))+'\n'+\
                r'$\sigma_{y}= $'+str(round(popt[4],3))+\
                r'$\,\pm\,$'+str(round(errs[4],3))+'\n',\
                    [1,1], xytext = (np.amax(x)+.1, np.amax(y)-5))
        plt.grid(which = 'minor')
        plt.legend()

        plt.show()

    return popt, errs
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def smoothing (vector, points = 3):
    """Smoothing macro.
                    - vector = input signal;
                    - points = smoothing range (odd).
    """
    if (int(points%2)==0):
        print ('\nOdd number of points needed. Smoothing NOT done.\n')
        return None
    else:
        smoothed = np.copy(vector)
        for i in range (points, len(vector)-points):
            smoothed[i] = 0.
            for j in range (-1*points/2+1,points/2+1):
                smoothed[i] += vector[i+j]/float(points)
        return smoothed
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def crossCorrCirc(f,g):
    """Cross-correlation function of the
        fluctuations of two signals, without padding, normalized.
        For flows: it will highligh a flow of particle f --> g;
                        - f = first signal;
                        - g = second signal.
        """

    N = len(f)

    F, G = fft(f), fft(g)

    cross = np.real(ifft(F*np.conj(G)))[:N]

    for i in range (len(cross)):
        cross[i] = cross[i]/(N-i)

    return cross
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def stics(f, mean_space, delay, duration):
    """ STICS normalized for time and space, with Heaviside in FT domain.
    """
    #
    # set up all the elements and parameters
    #
    T, M, N = f.shape
    cross = np.zeros((M, N))
    #
    # spatial correlation
    #

    for k in range(duration - delay):
        cross += np.real(\
            spatial_Xcorr_2(f[k, :, :], f[k+delay, :, :]))/\
                    mean_space[k]/mean_space[k+delay]

    return cross
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def kill_hot_pixels(matrix, value,file_name = None ):
    """ kills hor pixels in the image.
        Inputs:
        -matrix of the image
        -threshold value over which the average matrix value is assigned
        -flag to save the image and location of the imafe

    """
    matrix_2 = np.copy(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix[i,j]>=value):
                matrix_2[i,j] = np.average(matrix)
            else:
                matrix_2[i,j] = matrix[i,j]

    if (file_name != None):
        image = Image.fromarray(matrix_2)
        image.save(file_name)
    return matrix_2
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def kill_hot_pixels_bis(matrix, value, save='n', file_name='name.tif'):
    """ For visualization purposes. Check if average is ok.
    """
    matrix_2 = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix[i,j]<value): matrix_2[i,j] = matrix[i,j]
    if (save == 'y'):
        image = Image.fromarray(matrix_2)
        image.save(file_name)
    return matrix_2
#------------------------------------------------------------------------------
def centroid(image,threshold=0):
    """Calculation of cemtroids weighted on the illumination intensity (see eq 3.2)
    """
    # threshold image
    image[image<threshold] = 0.
    #meshgrid with subregions SH
    h,w = image.shape
    y = np.linspace(0,h-1,h)
    x = np.linspace(0,w-1,w)
    x,y = np.meshgrid(x,y)
    #Calculation of centroid coord (weighted on illumination see eq 3.2 report)
    avg_x = np.sum(x*image)/np.sum(image)
    avg_y = np.sum(y*image)/np.sum(image)

    return avg_x,avg_y
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def secondmoment(image,threshold=0):
    """Calculation of second moment of the image
    """
    # put as x y shape image
    centx,centy = centroid(image,threshold) # neglecting tip tilt
    h,w = image.shape
    y = np.linspace(0,h-1,h)-centy
    x = np.linspace(0,w-1,w)-centx
    x,y = np.meshgrid(x,y)
    second = np.sum(image*(x**2+y**2))/np.sum(image)

    return second


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#-------------------------------------------------------------------------------
def fftCrossCorr(im1,im2):
    """ Calulates the rosscorrelation of two images
    """
    im1fft=np.fft.fft2(im1)
    im2fft=np.fft.fft2(im2)
    return np.real(np.fft.fft2(np.conj(im1fft)*im2fft))

#-------------------------------------------------------------------------------
def realignStack(stack):
    """ Calulates the rosscorrelation of images in a stack
    """
    for i in range(stack.shape[0]-1):
        crosscorr=fftCrossCorr(stack[i+1,:,:]-np.mean(stack[i+1,:,:]),\
                                       stack[i,:,:]-np.mean(stack[0,:,:]))
        maxposy,maxposx=np.unravel_index(np.argmax(crosscorr),crosscorr.shape)
        if maxposy>crosscorr.shape[0]/2:
            maxposy=-(crosscorr.shape[0]-maxposy)
        if maxposx>crosscorr.shape[1]/2:
            maxposx=-(crosscorr.shape[1]-maxposx)
        stack[i+1,:,:]=np.roll(stack[i+1,:,:],-maxposy,axis=0)
        stack[i+1,:,:]=np.roll(stack[i+1,:,:],-maxposx,axis=1)

#-------------------------------------------------------------------------------
def stripes_pattern( gray_value, freq=16,height=1152,width=1920, show_image=False):
    """Returns a matrix which is a diffraction grating patthern.
       Inputs:
       -value of the stripe
       -period or frequency in pixels  of the diffraction grating NB it's doubled
       -height of the SLM
       -width of the SLM
       -Flag to show image on the screen
    """
    image=np.zeros((height,width), dtype=np.uint8)
    stripe=np.zeros((height,freq), dtype=np.uint8)
    stripe[:,0:int(freq/2)]=0
    stripe[:,int(freq/2):freq]=gray_value
    image=np.matlib.repmat(stripe,1,int(width/freq))
    if show_image==True:
        plt.figure('grating')
        plt.imshow(image)
        plt.show()

    return image
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def convert_pattern8bit(pattern,pattern_path=None,pattern_name=None,bmp=True ):
    """ Converts ndarray pattern into a 8 bit pattern
        If the flags pattern_path and pattern_name are activated it saves the
        converted pattern as npy file in the desired location
        If the flags pattern_path, pattern_name and bmp are activated it saves
        the converted pattern as bmp file in the desired location

        This function is useful to convert pattern generated with SLM controller
        to  be able to load them with SLM_screen or with the  BLINK software as
        bmp images.
    """
    pattern_8bit=((pattern-np.amin(pattern))/\
               (np.amax(pattern)-np.amin(pattern))*2**8).astype(np.uint8)

    if ( pattern_path != None ):
        np.save(pattern_path+'pattern_name',pattern_8bit)

    if ( bmp == True ):
       im=Image.fromarray(pattern_8bit)
       im.save(pattern_path+pattern_name+'.bmp')


    return pattern_8bit

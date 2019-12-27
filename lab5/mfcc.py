import numpy as np
import math


def wav2mstft(x, N, S, L):
    '''Chop a waveform into frames, and compute magnitude FFT of each frame.
    x = the signal (a nparray)
    N = length of the FFT
    S = frameskip
    L = framelength
    Returns: the positive-frequency components of the magnitude FFT of each frame (nparray),
      which is an array of size T by (N+1)/2.
    '''
    T = 1+int(math.ceil((len(x)-L)/S))
    X = np.zeros((T, 1+int(N/2)))
    for t in range(T):
        X[t,:] = np.absolute(np.fft.rfft(x[(t*S):min(t*S+L,len(x))], n=N))
    return(X)

def mstft2filterbank(X, Fs, Nm):
    '''Convert MSTFT into Filterbank coefficients in each frame
    X = the MSTFT of each frame (an nparray)
    Fs = the sampling frequency
    Nm = the number of mel-frequency bandpass filters to use
    Returns: the Filterbank coefficients in each frame (nparray)
    '''
    # First, create the triangle filters
    max_bin = X.shape[1]
    max_hertz = Fs/2
    max_mel = 1127*np.log(1+max_hertz/700.0)
    triangle_filters = np.zeros((max_bin, Nm))
    for m in range(Nm):
        corners=[int((max_bin*700.0/max_hertz)*(np.exp(max_mel*(m+n)/(1127*(Nm+1)))-1)) for n in range(3)]
        triangle_filters[corners[0]:corners[1],m]=np.linspace(0,1,corners[1]-corners[0],endpoint=False)
        triangle_filters[corners[1]:corners[2],m]=np.linspace(1,0,corners[2]-corners[1],endpoint=False)

    magnitudefilterbank = np.matmul(X,triangle_filters)
    maxcof = np.amax(magnitudefilterbank)
    return(np.log(np.maximum(1e-6*maxcof,magnitudefilterbank)))

    
def filterbank2mfcc(X, D):
    '''Convert from filterbank coefficients to MFCC.
    X = the filterbank coefficients of each frame (nframes X nbands)
    D = the number of features in the output (including static, delta, and delta-delta)
    Returns: mfcc (nparray)
    '''
    T = X.shape[0]
    Nm = X.shape[1]
    
    mfcc = np.zeros((T, D))  
    
    # Create the DCT transform matrix
    #Dover3 = int(np.ceil(D/3))
    #D = int(3*Dover3) # Make sure it's a multiple of 3
    dct_transform = np.zeros((Nm, D))
    for m in range(Nm):
        dct_transform[m,:] = np.cos(np.pi*(m+0.5)*np.arange(D)/Nm)
            
    # Create the MFCC
    mfcc = np.zeros((T, D))  
    mfcc = np.matmul(X,dct_transform)   # static MFCC
    #mfcc[1:T,Dover3:2*Dover3] = mfcc[1:T,0:Dover3] - mfcc[0:(T-1),0:Dover3]          # Delta-MFCC
    #mfcc[1:T,2*Dover3:D] = mfcc[1:T,Dover3:2*Dover3] - mfcc[0:(T-1),Dover3:2*Dover3]  # Delta-delta-MFCC
    return(mfcc)

    
def znorm(X):
    '''Mean and variance normalize the features.
    X = the filterbank coefficients of each frame (nframes X nbands)
    Returns: X, normalized so that each column has zero mean and unit standard deviation.
    '''
    return((X-np.average(X,axis=0))/np.std(X,axis=0))

                


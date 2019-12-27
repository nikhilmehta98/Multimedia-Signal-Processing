import numpy as np
import cmath,math

class Spectrograph(object):
    """Spectrograph: a device that computes a spectrogram."""
    def __init__(self,signal,samplerate,framelength,frameskip,numfreqs,maxfreq,dbrange):
        self.signal = signal   # A numpy array containing the samples of the speech signal
        self.samplerate = samplerate # Sampling rate [samples/second]
        self.framelength = framelength # Frame length [samples]
        self.frameskip = frameskip # Frame skip [samples]
        self.numfreqs = numfreqs # Number of frequency bins that you want in the spectrogram
        self.maxfreq = maxfreq # Maximum frequency to be shown, in Hertz
        self.dbrange = dbrange # All pixels that are dbrange below the maximum will be set to zero

    # PROBLEM 1.1
    #
    # Figure out how many frames there should be
    # so that every sample of the signal appears in at least one frame,
    # and so that none of the frames are zero-padded except possibly the last one.
    #
    # Result: self.nframes is an integer
    def set_nframes(self):
        self.nframes =  1+int((len(self.signal)-self.framelength)/self.frameskip)  # Not the correct value
        #
        # TODO: set self.nframes to something else

    # PROBLEM 1.2
    #
    # Chop the signal into overlapping frames
    # Result: self.frames is a numpy.ndarray, shape=(nframes,framelength), dtype='float64'
    def set_frames(self):
        self.frames = np.zeros((self.nframes,self.framelength),dtype='float64')
        #
        # TODO: fill self.frames
        for n in range(self.nframes):
            if (n*self.frameskip)+self.framelength < len(self.signal):
                self.frames[n] = self.signal[n*self.frameskip:(n*self.frameskip)+self.framelength]
            else:
                self.frames[n][0:len(self.signal[n*self.frameskip:])] = self.signal[n*self.frameskip:]

    # PROBLEM 1.3
    #
    # Window each frame with a Hamming window of the same length (use np.hamming)
    # Result: self.hammingwindow is a numpy.ndarray, shape=(framelength), dtype='float64'
    def set_hammingwindow(self):
        self.hammingwindow = np.hamming(self.framelength)
        #
        # TODO: fill self.hammingwindow
            
    # PROBLEM 1.4
    #
    # Window each frame with a Hamming window of the same length (use np.hamming)
    # Result: self.wframes is a numpy.ndarray, shape=(nframes,framelength), dtype='float64'
    def set_wframes(self):
        self.wframes = np.zeros(self.frames.shape, dtype='float64')
        #
        # TODO: fill self.wframes
        for n in range(len(self.frames)):
            self.wframes[n] = self.frames[n]*self.hammingwindow

    # PROBLEM 1.5
    #
    # Time alignment, in seconds, of the first sample of each frame, where signal[0] is at t=0.0
    # Result: self.timeaxis is a numpy.ndarray, shape=(nframes), dtype='float32'
    def set_timeaxis(self):
        self.timeaxis = np.zeros(self.nframes, dtype='float32')
        #
        # TODO: fill self.timeaxis    
        for n in range(len(self.frames)):
            self.timeaxis[n] = np.float32((n*self.frameskip)/self.samplerate)

    # PROBLEM 1.6
    #
    #   Length of the desired DFT.
    #   You want this to be long enough so that, in numfreqs bins, you get exactly maxfreq Hertz.
    #   result: self.dftlength is an integer
    def set_dftlength(self):
        self.dftlength = int(self.samplerate*self.numfreqs/self.maxfreq) # Not the correct value
        #
        # TODO: set self.dftlength

    # PROBLEM 1.7
    #
    # Compute the Z values (Z=exp(-2*pi*k*n*j/dftlength) that you will use in each DFT of the STFT.
    #    result (numpy array, shape=(numfreqs,framelength), dtype='complex128')
    #    result: self.zvalues[k,n] = exp(-2*pi*k*n*j/self.dftlength)
    def set_zvalues(self):
        self.zvalues = np.zeros((self.numfreqs,self.framelength), dtype='complex128')
        #
        # TODO: fill self.zvalues
        for n in range(self.numfreqs):
            for s in range(self.framelength):
                self.zvalues[n,s] = np.exp(-2*np.pi*s*n*1j/self.dftlength)

    # PROBLEM 1.8
    #
    # Short-time Fourier transform of the signal.
    #    result: self.stft is a numpy array, shape=(nframes,numfreqs), dtype='complex128'
    #    self.stft[m,k] = sum(wframes[m,:] * zvalues[k,:])
    def set_stft(self):
        self.stft = np.zeros((self.nframes,self.numfreqs), dtype='complex128')
        #
        # TODO: fill self.stft
        for n in range(self.nframes):
            for f in range(self.numfreqs):
                self.stft[n,f] = np.sum(self.wframes[n]*self.zvalues[f])

    # PROBLEM 1.9
    #
    # Find the level (in decibels) of the STFT in each bin.
    #    Normalize so that the maximum level is 0dB.
    #    Cut off small values, so that the lowest level is truncated to -60dB.
    #    result: self.levels is a numpy array, shape=(nframes,numfreqs), dtype='float64'
    #    self.levels[m,k] = max(-dbrange, 20*log10(abs(stft[m,k])/maxval))
    def set_levels(self):
        self.levels = np.zeros((self.nframes,self.numfreqs), dtype='float64')
        #
        # TODO: fill self.levels
        max_val = np.amax(abs(self.stft))
        for m in range(self.nframes):
            for k in range(self.numfreqs):
                self.levels[m,k] = max(-self.dbrange,20*np.log10(abs(self.stft[m,k])/max_val))

    # PROBLEM 1.10
    #
    # Convert the level-spectrogram into a spectrogram image:
    #    Add 60dB (so the range is from 0 to 60dB), scale by 255/60 (so the max value is 255),
    #    and convert to data type uint8.
    #    result: self.image is a numpy array, shape=(nframes,numfreqs), dtype='uint8'
    def set_image(self):
        self.image = np.zeros((self.nframes,self.numfreqs), dtype='uint8')
        #
        # TODO: fill self.image
        for m in range(self.nframes):
            for k in range(self.numfreqs):
                self.image[m,k] = np.uint8((self.levels[m,k] + 60) * (255/60))


import numpy as np
import cmath,math,os,collections
from PIL import Image

steps = [
    'ypbpr',
    'rowconv',
    'gradient',
    'background',
    'clipped',
    'matchedfilters',
    'matches',
    'features',
    'confusion',
    'accuracy'
    ]
    

class Dataset(object):
    """
    dataset=Dataset(classes), where classes is a list of class names.
    Result: 
    dataset.data is a list of observation data read from the files,
    dataset.labels gives the class number of each datum (between 0 and len(dataset.classes)-1)
    dataset.classes is a copy of the provided list of class names --- there should be exactly two.
    """
    def __init__(self,classes,nperclass):
        # Number of sets (train vs. test), number of classes (always 2), num toks per set per split (6)
        self.nclasses = 2
        self.nperclass = nperclass
        self.ndata = self.nclasses * self.nperclass
        # Load classes from the input.  If there are more than 2, only the first 2 are used
        self.classes = classes
        # Number of rows per image, number of columns, number of colors
        self.nrows = 200
        self.ncols = 300
        self.ncolors = 3
        # Data sets, as read from the input data directory
        self.labels = np.zeros((self.ndata),dtype='int')
        self.data = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata),dtype='float64')
        for label in range(0,self.nclasses):
            for num in range(0,self.nperclass):
                datum = label*(self.nperclass) + num
                filename = os.path.join('data','%s%2.2d.jpg'%(self.classes[label],num+1))
                self.labels[datum] = label
                self.data[:,:,:,datum] = np.asarray(Image.open(filename))
        
    # PROBLEM 3.0
    #
    # Convert image into Y-Pb-Pr color space, using the ITU-R BT.601 conversion
    #   [Y;Pb;Pr]=[0.299,0.587,0.114;-0.168736,-0.331264,0.5;0.5,-0.418688,-0.081312]*[R;G;B].
    # Put the results into the numpy array self.ypbpr[:,:,:,m]
    def set_ypbpr(self):
        self.ypbpr = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata))
        #
        # TODO: convert each RGB image to YPbPr
        for m in range(self.ndata):
            self.ypbpr[:,:,0,m] = 0.299*self.data[:,:,0,m] + 0.587*self.data[:,:,1,m] + 0.114*self.data[:,:,2,m]
            self.ypbpr[:,:,1,m] = -0.168736*self.data[:,:,0,m] + -0.331264*self.data[:,:,1,m] + 0.5*self.data[:,:,2,m]
            self.ypbpr[:,:,2,m] = 0.5*self.data[:,:,0,m] + -0.418688*self.data[:,:,1,m] + -0.081312*self.data[:,:,2,m]

    # PROBLEM 3.1
    #
    # Filter each row of ypbpr with two different filters.
    # The first filter is the difference: [1,0,-1]
    # The second filter is the average: [1,2,1]
    # Keep only the 'valid' samples, thus the result has size (nrows,ncols-2,2*ncolors)
    # The output 'colors' are (diffY,diffPb,diffPr,aveY,avePb,avePr).
    def set_rowconv(self):
        self.rowconv = np.zeros((self.nrows,self.ncols-2,2*self.ncolors,self.ndata))
        # TODO: calculate the six output planes (diffY,diffPb,diffPr,aveY,avePb,avePr).
        for m in range(self.ndata):
            for i in range(self.nrows):
                for j in range(self.ncols-2):
                    self.rowconv[i,j,0:3,m] = -self.ypbpr[i,j,:,m] + self.ypbpr[i,j+2,:,m]
                    self.rowconv[i,j,3:6,m] = self.ypbpr[i,j,:,m] + 2*self.ypbpr[i,j+1,:,m] + self.ypbpr[i,j+2,:,m]
                    
    # PROBLEM 3.2
    #
    # Calculate the (Gx,Gy) gradients of the YPbPr images using Sobel mask.
    # This is done by filtering the columns of self.rowconv.
    # The first three "colors" are filtered by [1,2,1] in the columns.
    # The last three "colors" are filtered by [1,0,-1] in the columns.
    # Keep only 'valid' outputs, so size is (nrows-2,ncols-2,2*ncolors)
    def set_gradient(self):
        self.gradient = np.zeros((self.nrows-2,self.ncols-2,2*self.ncolors,self.ndata))
        # TODO: compute the Gx and Gy planes of the Sobel features for each image
        for m in range(self.ndata):
            for j in range(self.ncols-2):
                for i in range(self.nrows-2):
                    self.gradient[i,j,0:3,m] = self.rowconv[i,j,0:3,m] + 2*self.rowconv[i+1,j,0:3,m] + self.rowconv[i+2,j,0:3,m]
                    self.gradient[i,j,3:6,m] = -self.rowconv[i,j,3:6,m] + self.rowconv[i+2,j,3:6,m]
        

    # PROBLEM 3.3
    #
    # Create a matched filter, for each class, by averaging the YPbPr images of that class,
    # removing the first two rows, last two rows, first two columns, and last two columns,
    # flipping left-to-right, and flipping top-to-bottom.
    def set_matchedfilters(self):
        self.matchedfilters = np.zeros((self.nrows-4,self.ncols-4,self.ncolors,self.nclasses))
        # TODO: for each class, average the YPbPr images, fliplr, and flipud
        for n in range(self.nclasses):
            for m in range(self.nperclass):
                self.matchedfilters[:,:,:,n] += self.ypbpr[2:-2,2:-2,:,n*self.nperclass+m]/self.nperclass
            self.matchedfilters[:,:,0,n] = np.fliplr(np.flipud(self.matchedfilters[:,:,0,n]))
            self.matchedfilters[:,:,1,n] = np.fliplr(np.flipud(self.matchedfilters[:,:,1,n]))
            self.matchedfilters[:,:,2,n] = np.fliplr(np.flipud(self.matchedfilters[:,:,2,n]))

    # PROBLEM 3.4
    #
    # self.matches[:,:,c,d,z] is the result of filtering self.ypbpr[:,:,c,d]
    #   with self.matchedfilters[:,:,c,z].  Since we're not using scipy, you'll have to
    #   implement 2D convolution yourself, for example, by convolving each row, then adding
    #   the results; or by just multiplying and adding at each shift.
    def set_matches(self):
        self.matches = np.zeros((5,5,self.ncolors,self.ndata,self.nclasses))
        # TODO: compute 2D convolution of each matched filter with each YPbPr image
        for n in range(self.nclasses):
            for m in range(self.ndata):
                for i in range(5):
                    for j in range(5):
                        
                        self.matches[i,j,0,m,n] = np.sum(np.multiply(self.ypbpr[i:self.nrows-(4-i),j:self.ncols-(4-j),0,m],np.fliplr(np.flipud(self.matchedfilters[:,:,0,n]))))
                        self.matches[i,j,1,m,n] = np.sum(np.multiply(self.ypbpr[i:self.nrows-(4-i),j:self.ncols-(4-j),1,m],np.fliplr(np.flipud(self.matchedfilters[:,:,1,n]))))
                        self.matches[i,j,2,m,n] = np.sum(np.multiply(self.ypbpr[i:self.nrows-(4-i),j:self.ncols-(4-j),2,m],np.fliplr(np.flipud(self.matchedfilters[:,:,2,n]))))

    # PROBLEM 3.5 
    #
    # Create a feature vector from each image, showing three image features that
    # are known to each be useful for some problems.
    # self.features[d,0] is norm(Pb)-norm(Pr)
    # self.features[d,1] is norm(Gx[luminance])-norm(Gy[luminance])
    # self.features[d,2] is norm(match to class 1[all colors]) - norm(match to class 0[all colors])
    def set_features(self):
        self.features = np.zeros((self.ndata,3))
        #
        # TODO: Calculate color feature, gradient feature, and matched filter feature for each image
        for m in range(self.ndata):
            self.features[m,0] = np.linalg.norm(self.ypbpr[:,:,1,m])-np.linalg.norm(self.ypbpr[:,:,2,m])
            self.features[m,1] = np.linalg.norm(self.gradient[:,:,0,m])-np.linalg.norm(self.gradient[:,:,3,m])
            self.features[m,2] = np.linalg.norm(self.matches[:,:,:,m,1])-np.linalg.norm(self.matches[:,:,:,m,0])

    # PROBLEM 3.6
    #
    # self.accuracyspectrum[d,f] = training corpus accuracy of the following classifier:
    #   if self.features[k,f] >= self.features[d,f], then call datum k class 1, else call it class 0.
    def set_accuracyspectrum(self):
        self.accuracyspectrum = np.zeros((self.ndata,3))
        #
        # TODO: Calculate the accuracy of every possible single-feature classifier
        accuracy = np.zeros((self.ndata,self.ndata,3))
        for m in range(self.ndata):
            for f in range(3):
                for n in range(self.ndata):
                    if self.features[m,f] >= self.features[n,f]:
                        accuracy[m,n,f] = 1
        for m in range(self.ndata):
            for f in range(3):
                count = 0
                for n in range(self.ndata):
                    if accuracy[n,m,f] == self.labels[n]:
                        count += 1
                self.accuracyspectrum[m,f] = count/self.ndata

    # PROBLEM 3.7
    #
    # self.bestclassifier specifies the best classifier for each feature
    # self.bestclassifier[f,0] specifies the best threshold for feature f
    # self.bestclassifier[f,1] specifies the polarity:
    #   to specify that class 1 has feature >= threshold, set self.bestclassifier[f,1] = +1
    #   to specify that class 1 has feature < threshold, set self.bestclassifier[f,1] = -1
    # self.bestclassifier[f,2] gives the accuracy of the best classifier for feature f,
    #   computed from the accuracyspectrum as
    #   accuracy[f] = max(max(accuracyspectrum[:,f]), 1-min(accuracyspectrum[:,f])).
    #   If the max is selected, then polarity=+1; if the min is selected, then polarity=-1.
    def set_bestclassifier(self):
        self.bestclassifier = np.zeros((3,3))
        #
        # TODO: find the threshold and polarity of best classifier for each feature
        for f in range(3):
            self.bestclassifier[f,0] = self.features[np.argmax(self.accuracyspectrum[:,f]),f]
            if max(self.accuracyspectrum[:,f]) < 1-min(self.accuracyspectrum[:,f]):
                self.bestclassifier[f,1] = -1
                self.bestclassifier[f,2] = 1-min(self.accuracyspectrum[:,f])
            else:
                self.bestclassifier[f,1] = 1
                self.bestclassifier[f,2] = max(self.accuracyspectrum[:,f])
            
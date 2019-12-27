import numpy as np
import cmath,math,os,collections
from PIL import Image

people = ['Arnold_Schwarzenegger','George_HW_Bush','George_W_Bush','Jiang_Zemin']

class KNN(object):
    """KNN: a class that computes K-nearest-neighbors of each image in a dataset"""
    def __init__(self,datadir,nfeats,transformtype,K):
        self.nfeats = nfeats  # Number of features to keep
        self.transformtype = transformtype # Type of transform, 'dct' or 'pca'
        self.K = K # number of neighbors to use in deciding the class label of each token
        self.npeople = 4
        self.nimages = 12
        self.ndata = self.npeople*self.nimages
        self.nrows = 64
        self.ncols = 64
        self.npixels = self.nrows * self.ncols
        self.data = np.zeros((self.ndata,self.nrows,self.ncols),dtype='float64')
        self.labels = np.zeros(self.ndata, dtype='int')
        for person in range(0,self.npeople):
            for num in range(0,self.nimages):
                datum = 12*person + num
                datafile = os.path.join(datadir,'%s_%4.4d.ppm'%(people[person],num+1))
                img = np.asarray(Image.open(datafile))
                bw_img = np.average(img,axis=2)
                self.data[datum,:,:] = bw_img
                self.labels[datum] = person
        
    # PROBLEM 2.0
    #
    # set_vectors - reshape self.data into self.vectors.
    #   Vector should scan the image in row-major ('C') order, i.e.,
    #   self.vectors[datum,n1*ncols+n2] = self.data[datum,n1,n2]
    def set_vectors(self):
        self.vectors = np.zeros((self.ndata,self.nrows*self.ncols),dtype='float64')
        #
        # TODO: fill self.vectors
        for i in range(self.ndata):
            flat = np.zeros(self.npixels)
            for n1 in range(self.nrows):
                for n2 in range(self.ncols):
                    flat[n1*self.ncols+n2] = self.data[i,n1,n2]
            self.vectors[i] = flat

    # PROBLEM 2.1
    #
    # set_mean - find the global mean image vector
    def set_mean(self):
        self.mean = np.zeros(self.npixels, dtype='float64')
        # TODO: fill self.mean
        for p in range(self.npixels):
            for i in range(self.ndata):
                self.mean[p] = self.mean[p] + self.vectors[i,p]
            self.mean[p] = self.mean[p]/self.ndata

    # PROBLEM 2.2
    #
    # set_centered - compute the zero-centered dataset, i.e., subtract the mean from each vector.
    def set_centered(self):
        self.centered = np.zeros((self.ndata,self.npixels), dtype='float64')
        # TODO: fill self.centered
        for i in range(self.npixels):
            self.centered[:,i] = self.vectors[:,i]-self.mean[i]
            
    # PROBLEM 2.3
    #
    # set_transform - compute the feature transform matrix (DCT or PCA)
    #  If transformtype=='dct':
    #    transform[ktot,ntot] = basis[k1,n1,nrows] * basis[k2,n2,ncols]
    #    basis[k1,n1,nrows] = (D/sqrt(nrows)) * cos(pi*(n1+0.5)*k1/nrows)
    #    D = 1 if k1==0, otherwise D = sqrt(2)
    #    Image pixels are scanned in row-major order ('C' order), thus ntot = n1*ncols + n2.
    #    Frequencies are scanned in diagonal L2R order: (k1,k2)=(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),...
    #  If transformtype=='pca':
    #    self.transform[k,:] is the unit-norm, positive-first-element basis-vector in the
    #       principal component direction with the k'th highest eigenvalue.
    #    You can get these from eigen-analysis of covariance or gram, or by SVD of the data matrix.
    #    To pass the autograder, you must check the sign of self.transform[k,0], for each k,
    #    and set self.transform[k,:] = -self.transform[k,:] if self.transform[k,0] < 0.
    def set_transform(self):
        def basis(k,n,i):
            if k == 0:
                D = 1
            else:
                D = np.sqrt(2)
            x = (D/np.sqrt(i)) * np.cos(np.pi*(n+0.5)*k/i)
            return x
        if self.transformtype=='dct':
            self.transform = np.zeros((self.nfeats,self.npixels),dtype='float64')
            # TODO: set self.transform in the DCT case
            k_vals = []
            axis = int(np.sqrt(self.nfeats))
            for line in range(1,int(2*np.sqrt(self.nfeats))):
                start_col = max(0, line - axis)
                count = min(line, (axis - start_col), axis)
                for j in range(0, count) :
                    k_vals.append([min(axis, line) - j - 1,start_col + j])
#             print(len(k_vals))
            for i, k in enumerate(k_vals):
                for n1 in range(self.nrows):
                    for n2 in range(self.ncols):
                        self.transform[i,n1*self.ncols + n2] = basis(k[0],n1,self.nrows)*basis(k[1],n2,self.ncols)
        elif self.transformtype=='pca':
            self.transform = np.zeros((self.nfeats,self.npixels),dtype='float64')
            # TODO: set self.transform in the PCA case
            u, s, v = np.linalg.svd(self.centered)
            for i in range(self.nfeats):
                self.transform[i] = v[i]
                if self.transform[i,0] < 0:
                    self.transform[i] = -self.transform[i]

    # PROBLEM 2.4
    #
    # set_features - transform the centered dataset to generate feature vectors.
    def set_features(self):
        self.features = np.zeros((self.ndata,self.nfeats),dtype='float64')
        # TODO: fill self.features
        self.features = np.dot(self.centered,np.transpose(self.transform))

    # PROBLEM 2.5
    #
    # set_energyspectrum: the fraction of total centered-dataset variance explained, cumulatively,
    #   by all feature dimensions up to dimension k, for 0<=k<nfeats.
    def set_energyspectrum(self):
        self.energyspectrum = np.zeros(self.nfeats,dtype='float64')
        #
        # TODO: calculate total dataset variances, then set self.energyspectrum
        E_tot = 0
        for i in range(self.ndata):
            E_tot += np.sum(np.absolute(self.centered[i])**2)
        self.energyspectrum[0] = np.sum(np.absolute(self.features[:,0])**2)
        for i in range(1,self.nfeats):
            self.energyspectrum[i] = self.energyspectrum[i-1] + np.sum(np.absolute(self.features[:,i])**2)
        self.energyspectrum /= E_tot
    
    # PROBLEM 2.6
    #
    # set_neighbors - indices of the K nearest neighbors of each feature vector (not including itself).
    #    return: a matrix of datum indices, i.e.,
    #    self.features[self.neighbors[n,k],:] should be the k'th closest vector to self.features[n,:].
    def set_neighbors(self):
        self.neighbors = np.zeros((self.ndata,self.K), dtype='int')
        #
        # TODO: fill self.neighbors
        for n, feat in enumerate(self.features):
            ignore = [n]
            for k in range(self.K):
                closest = 0
                cur_dist = np.inf
                for i, f in enumerate(self.features):
                    if i in ignore:
                        continue
                    dist = np.linalg.norm(feat-f)
                    if dist < cur_dist:
                        closest = i
                        cur_dist = dist
                self.neighbors[n,k] = closest
                ignore.append(closest)
            
    # PROBLEM 2.7
    #
    # set_hypotheses - K-nearest-neighbors vote, to choose a hypothesis person label for each datum.
    #   If K>1, then check for ties!  If the vote is a tie, then back off to 1NN among the tied options.
    #   In other words, among the tied options, choose the one that has an image vector closest to
    #   the datum being tested.
    def set_hypotheses(self):
        self.hypotheses = np.zeros(self.ndata, dtype='int')
        #
        # TODO: fill self.hypotheses
        labels = []
        map_tot = []
        for n, nbors in enumerate(self.neighbors):
            l = []
            n = []
            map1 = {}
            for i, nbor in enumerate(nbors):
                l.append(self.labels[nbor])
                if (self.labels[nbor] in map1):
                    map1[self.labels[nbor]].append(nbor)
                else:
                    map1[self.labels[nbor]] = [nbor]
                n.append(nbor)
            labels.append([l,n])
            map_tot.append(map1)
#         print(labels)
#         print (map_tot)
        for n, group in enumerate(labels):
            occurs = np.zeros(self.npeople,dtype='int')
            for l in group[0]:
                occurs[l] += 1
            occurs = occurs.tolist()
            max_occurs = max(occurs)
            if occurs.count(max_occurs) == 1:
                self.hypotheses[n] = occurs.index(max_occurs)
            else:
                closest = occurs.index(max_occurs)
#                 print(closest)
                for item in range(closest,self.npeople):
                    if (occurs[item] == 0):
                        continue
                    if occurs[item] == max_occurs:
                        check_old = map_tot[n][closest][0]
                        check_new = map_tot[n][item][0]
#                         print(check)
                        if list(group[1]).index(check_new) < list(group[1]).index(check_old):
                            closest = item
                self.hypotheses[n] = closest                
        

    # PROBLEM 2.8
    #
    # set_confusion - compute the confusion matrix
    #   confusion[r,h] = number of images of person r that were classified as person h    
    def set_confusion(self):
        self.confusion = np.zeros((self.npeople,self.npeople), dtype='int')
        #
        # TODO: fill self.confusion
        for r in range(self.npeople):
            for h in range(self.npeople):
                for i in range(len(self.hypotheses)):
                    if(self.hypotheses[i] == h and self.labels[i] == r):
                        self.confusion[r,h] += 1
                
    # PROBLEM 2.9
    #
    # set_metrics - set self.metrics = [ accuracy, recall, precision ]
    #   recall is the average, across all people, of the recall rate for that person
    #   precision is the average, across all people, of the precision rate for that person
    def set_metrics(self):
        self.metrics = [0.0, 1.0, 1.0]  # probably not the correct values!
        #
        # TODO: fill self.metrics
        for i in range(self.ndata):
            if (self.hypotheses[i] == self.labels[i]):
                self.metrics[0] += 1
        self.metrics[0] /= self.ndata
            
        p_a = []
        for p in range(self.npeople):
            count = 0
            correct = 0
            for i in range(self.ndata):
                if self.labels[i] == p:
                    count += 1
                    if self.hypotheses[i] == p:
                        correct += 1
            p_a.append(correct/count)
        self.metrics[1] = np.average(p_a)
        
        p_a = []
        for p in range(self.npeople):
            count = 0
            correct = 0
            for i in range(self.ndata):
                if self.hypotheses[i] == p:
                    count += 1
                    if self.labels[i] == p:
                        correct += 1
            p_a.append(correct/count)
        self.metrics[2] = np.average(p_a)
        
        print (self.metrics)
        
    # do_all_steps:
    #   This function is given, here, in case you want to do all of the steps all at once.
    #   To do that, you can type
    #   knn=KNN('data',36,'dct',4)
    #   knn.do_all_steps()
    #
    def do_all_steps(self):
        self.set_vectors()
        self.set_mean()
        self.set_centered()
        self.set_transform()
        self.set_features()
        self.set_energyspectrum()
        self.set_neighbors()
        self.set_hypotheses()
        self.set_confusion()
        self.set_metrics()


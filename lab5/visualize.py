import os,submitted,matplotlib,matplotlib.pyplot,math,wave,struct
import numpy as np

def mag2level(X):
    clip = 1e-3 * np.amax(X)
    if clip==0:
        return(np.zeros(X.shape))
    else:
        return(20*np.log10(np.maximum(clip,X)))

# plotter
class Plotter(object):
    '''Generate a series of plots that are useful for debugging your submitted code.'''
    def __init__(self,outputdirectory):
        '''Create the specified output directory, and initialize plotter to put its plots there'''
        os.makedirs(outputdirectory,exist_ok=True)
        self.figs = [ matplotlib.pyplot.figure(n) for n in range(0,len(submitted.steps)) ]
        self.outputdirectory = outputdirectory

    def make_plots(self, epoch):
        '''Create a new dataset object, run all the steps, and make all the plots'''
        self.output_filename=os.path.join(self.outputdirectory,'epoch%d'%(epoch))
        self.dataset = submitted.Dataset(epoch)
        self.sf = [ self.dataset.startframe[n] for n in (0,1) ]        
        self.ef = [ self.dataset.endframe[n] for n in (0,1) ]        
        self.st = [ self.dataset.starttok[n] for n in (0,1) ]        
        self.et = [ self.dataset.endtok[n] for n in (0,1) ]        
        self.sgram = [ mag2level(self.dataset.mstft[self.sf[n]:self.ef[n]]) for n in (0,1) ]
        for n in range(0,len(submitted.steps)):
            print('Creating plots for epoch %d step %d (%s)'%(epoch,n,submitted.steps[n]))
            getattr(self.dataset, 'set_' + submitted.steps[n])()
            plotter_method = getattr(self, 'plot_%s'%(submitted.steps[n]))
            self.figs[n].clf()
            plotter_method(self.figs[n], self.dataset)                
            self.figs[n].savefig(self.output_filename + '_step%d.png'%(n))
        
    def plot_surprisal(self, f, dataset):
        aa = f.subplots(nrows=4,ncols=1)
        aa[0].imshow(np.flipud(self.sgram[0].T),cmap='gray',aspect='auto')
        aa[0].set_title('%s spectrogram, filterbank, mfcc, surprisal'%(dataset.utts[0]))
        aa[1].imshow(np.flipud(dataset.filterbank[self.sf[0]:self.ef[0],:].T),cmap='gray',aspect='auto')
        aa[2].imshow(np.flipud(dataset.mfcc[self.sf[0]:self.ef[0],:].T),cmap='gray',aspect='auto')
        aa[3].imshow(np.flipud(mag2level(dataset.surprisal[self.sf[0]:self.ef[0],self.st[0]:self.et[0]]).T),aspect='auto')
        aa[3].set_yticks(np.arange(self.et[0]-self.st[0]))
        aa[3].set_yticklabels(dataset.toks[self.st[0]:self.et[0]])

    def plot_alphahat(self, f, dataset):
        aa = f.subplots(nrows=4,ncols=1)
        for u in (0,1):
            aa[2*u].imshow(np.flipud(self.sgram[u].T),cmap='gray',aspect='auto')
            aa[2*u+1].imshow(dataset.alphahat[self.sf[u]:self.ef[u],0:dataset.ntoks[u]].T,aspect='auto')
            aa[2*u+1].set_yticks(np.arange(dataset.ntoks[u]))
            aa[2*u+1].set_yticklabels(dataset.toks[self.st[u]:self.et[u]])
            
        aa[0].set_title('%s: spectrogram, alphahat'%(dataset.utts[0]))
        aa[3].set_xlabel('%s: spectrogram, alphahat'%(dataset.utts[1]))
        
    def plot_betahat(self, f, dataset):
        aa = f.subplots(nrows=4,ncols=1)
        for u in (0,1):
            aa[2*u].imshow(np.flipud(self.sgram[u].T),cmap='gray',aspect='auto')
            aa[2*u+1].imshow(dataset.betahat[self.sf[u]:self.ef[u],0:dataset.ntoks[u]].T,aspect='auto')
            aa[2*u+1].set_yticks(np.arange(self.et[u]-self.st[u]))
            aa[2*u+1].set_yticklabels(dataset.toks[self.st[u]:self.et[u]])
        aa[0].set_title('%s: spectrogram, log betahat'%(dataset.utts[0]))
        aa[3].set_xlabel('%s: spectrogram, log betahat'%(dataset.utts[1]))
        
    def plot_gamma(self, f, dataset):
        aa = f.subplots(nrows=4,ncols=1)
        for u in (0,1):
            aa[2*u].imshow(np.flipud(self.sgram[u].T),cmap='gray',aspect='auto')
            #qstar = np.argmax(dataset.gamma[self.sf[u]:self.ef[u],:],axis=1)
            aa[2*u+1].imshow(dataset.gamma[self.sf[u]:self.ef[u],0:dataset.ntoks[u]].T,aspect='auto')
            #aa[2*u+1].plot(qstar)
            aa[2*u+1].set_yticks(np.arange(self.et[u]-self.st[u]))
            aa[2*u+1].set_yticklabels(dataset.toks[self.st[u]:self.et[u]])
        aa[0].set_title('%s: spectrogram, gamma'%(dataset.utts[0]))
        aa[3].set_xlabel('%s: spectrogram, gamma'%(dataset.utts[1]))
        
    def plot_xi(self, f, dataset):
        aa = f.subplots(nrows=4,ncols=1)
        for u in (0,1):
            aa[2*u].imshow(np.flipud(self.sgram[u].T),cmap='gray',aspect='auto')
            aa[2*u+1].imshow(np.sum(dataset.xi[self.sf[u]:self.ef[u],0:dataset.ntoks[u],0:dataset.ntoks[u]],axis=1).T,aspect='auto')
            aa[2*u+1].set_yticks(np.arange(self.et[u]-self.st[u]))
            aa[2*u+1].set_yticklabels(dataset.toks[self.st[u]:self.et[u]])
        aa[0].set_title('%s: spectrogram, xi'%(dataset.utts[0]))
        aa[3].set_xlabel('%s: spectrogram, xi'%(dataset.utts[1]))
        
    def plot_mu(self, f, dataset):
        aa = f.subplots(nrows=2,ncols=2)
        for r in range(2):
            for c in range(2):
                i = 2*c+r # tok index
                if c==1 and r==1:
                    i = 9
                aa[r,c].plot(dataset.mu[dataset.tok2type[i],:])
                aa[r,c].set_title('mean /%s/'%(dataset.toks[i]))
        
    def plot_var(self, f, dataset):
        aa = f.subplots(nrows=2,ncols=2)
        for r in range(2):
            for c in range(2):
                i = 2*c+r # tok index
                if c==1 and r==1:
                    i = 9
                aa[r,c].plot(dataset.var[i,:])
                aa[r,c].set_title('var /%s/'%(dataset.toks[i]))
        
    def plot_tpm(self, f, dataset):
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.tpm,aspect='auto')
        a.set_title('Transition probability matrix after epoch %d'%(dataset.epoch))
        a.set_xticks(np.arange(dataset.ntypes))
        a.set_yticks(np.arange(dataset.ntypes))
        a.set_xticklabels(dataset.model['phones'])
        a.set_yticklabels(dataset.model['phones'])

    def plot_trainedmodel(self, f, dataset):
        pass
    
#####################################################
# If this is called from the command line,
# create a plotter with the specified arguments (or with default arguments),
# then create the corresponding plots in the 'make_cool_plots_outputs' directory.
if __name__ == '__main__':
    
    plotter=Plotter('vis')
    for epoch in range(2):
        plotter.make_plots(epoch)
    

import os,submitted,matplotlib.pyplot
import numpy as np
import cv2,subprocess

# plotter
class Plotter(object):
    '''Generate plots and video that are useful for debugging your submitted code.'''
    def __init__(self,outputdirectory):
        '''Create the specified output directory, and initialize plotter to put its plots there'''
        os.makedirs(outputdirectory,exist_ok=True)
        self.figs = [ matplotlib.pyplot.figure(n) for n in range(0,len(submitted.steps)) ]
        self.outputdirectory = outputdirectory

    def make_plots(self, testcase):
        '''Create a new dataset object, run all the steps, and make all the plots'''
        self.output_filename=os.path.join(self.outputdirectory,'testcase%d'%(testcase))
        self.dataset = submitted.Dataset(testcase)
        for n in range(0,len(submitted.steps)):
            print('Creating plots for testcase %d step %d (%s)'%(testcase,n,submitted.steps[n]))
            getattr(self.dataset, 'set_' + submitted.steps[n])()
            plotter_method = getattr(self, 'plot_%s'%(submitted.steps[n]))
            self.figs[n].clf()
            plotter_method(self.figs[n], self.dataset)                
            self.figs[n].savefig(self.output_filename + '_step%d.png'%(n))
        
    def plot_layer1validate(self, f, dataset):
        '''Create a plot of the layer1 hidden weights as a function of time'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.layer1validate.T)
        a.set_title('Layer 1 activations, validation data')
        a.set_xlabel('Frame index')
        a.set_ylabel('Node index')

    def plot_layer1test(self, f, dataset):
        '''Create a plot of the layer1 hidden weights as a function of time'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.layer1test.T)
        a.set_title('Layer 1 activations, test data')
        a.set_xlabel('Frame index')
        a.set_ylabel('Node index')

    def plot_layer2validate(self, f, dataset):
        '''Create a plot of the layer2 hidden weights as a function of time'''
        aa = f.subplots(nrows=3,ncols=1,sharex=True)
        aa[0].plot(dataset.layer2validate[:,0])
        aa[0].set_title('Width, Validation data')
        aa[1].plot(dataset.layer2validate[:,1])
        aa[1].set_title('Height0')
        aa[2].plot(dataset.layer2validate[:,2])
        aa[2].set_title('Height1')
        aa[2].set_xlabel('Frame index')
        
    def plot_layer2test(self, f, dataset):
        '''Create a plot of the layer2 hidden weights as a function of time'''
        aa = f.subplots(nrows=3,ncols=1,sharex=True)
        aa[0].plot(dataset.layer2test[:,0])
        aa[0].set_title('Width, Test data')
        aa[1].plot(dataset.layer2test[:,1])
        aa[1].set_title('Height0')
        aa[2].plot(dataset.layer2test[:,2])
        aa[2].set_title('Height1')
        aa[2].set_xlabel('Frame index')
        
    def plot_grad2validate(self, f, dataset):
        '''Create a plot of the grad2 gradient after one epoch of training'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.grad2validate)
        a.set_title('Layer 2 weights gradient')
        a.set_xlabel('Hidden node number')
        a.set_ylabel('Output node number')

    def plot_grad1validate(self, f, dataset):
        '''Create a plot of the grad1 gradient'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.grad1validate)
        a.set_title('Layer 1 weights gradient')
        a.set_xlabel('Input node number')
        a.set_ylabel('Hidden node number')

    def plot_corners(self, f, dataset):
        '''Create plot showing how the corners of one triangle change over time'''
        aa = f.subplots(nrows=2,ncols=1,sharex=True)
        labels='xy'
        for row in (0,1):
            aa[row].plot(dataset.corners[:,0,:,row])
            aa[row].set_title(labels[row]+'-coordinates, corners of triangle 0')
        aa[1].set_xlabel('Frame index')

    def plot_cartesian2barycentric(self, f, dataset):
        '''Create plot showing the determinants of the transforms, as a function of frame number'''
        a = f.add_subplot(1,1,1)
        dets = [[ np.linalg.det(dataset.cartesian2barycentric[t,k,:,:]) for k in range(5) ] for t in range(dataset.ntest) ]
        a.plot(dets)
        a.set_title('Determinants of the first five cartesian2barycentric transforms')
        a.set_xlabel('Frame index')
        
    def plot_barycentric(self, f, dataset):
        '''Create plots showing Barycentric coordinates of four different pixels'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.barycentric[:,:,dataset.ncols//2,0,0].T)
        a.set_title('First Barycentric coordinate, first triangle, for center column of image')
        a.set_xlabel('Frame ID')
        a.set_ylabel('Row ID')
        
    def plot_pix2tri(self, f, dataset):
        '''Plot pix2tri along the column at the center of the image'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.pix2tri[:,:,dataset.ncols//2].T)
        a.set_title('pix2tri of column at the center of the image')
        a.set_xlabel('Frame number')
        a.set_ylabel('Row number')
        
    
    def plot_refcoordinate(self, f, dataset):
        '''Plot y shifts along the column at the center of the image, as function of frame index'''
        a = f.add_subplot(1,1,1)
        a.plot(dataset.refcoordinate[:,:,dataset.ncols//2,1])
        a.set_title('refimage Y coordinates of pixels in center column')
        a.set_xlabel('Frame number')
        a.set_ylabel('Row number')
        
    
    def plot_intensities(self, f, dataset):
        '''Use OpenCV to write out the complete video, and if available, use ffmpeg to add audio'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.refimage,cmap='gray')
        video_filename = self.output_filename + '_video.avi'
        out = cv2.VideoWriter(video_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (dataset.ncols,dataset.nrows), 0)

        for frame in range(dataset.ntest):
            out.write(dataset.intensities[frame,:,:].astype(np.uint8))
        out.release()

        audio_filename=os.path.join('data','test%d.wav'%(dataset.testcase))
        audiovisual=self.output_filename + '_audiovisual.avi'
        cmd=['ffmpeg','-i',video_filename,'-i',audio_filename,'-codec','copy','-shortest',audiovisual]
        subprocess.call(cmd)
        
#####################################################
# If this is called from the command line,
# create a plotter with the specified arguments (or with default arguments),
# then create the corresponding plots in the 'make_cool_plots_outputs' directory.
if __name__ == '__main__':
    
    plotter=Plotter('vis')
    for testcase in range(5):
        plotter.make_plots(testcase)
    

import os,submitted,matplotlib,matplotlib.pyplot
import numpy as np

# plotter
class Plotter(object):
    '''Generate a series of plots that are useful for debugging your submitted code.'''
    def __init__(self,outputdirectory):
        '''Create the specified output directory, and initialize plotter to put its plots there'''
        os.makedirs(outputdirectory,exist_ok=True)
        self.figs = [ matplotlib.pyplot.figure(n) for n in range(0,len(submitted.steps)) ]
        self.outputdirectory = outputdirectory
        self.tmin=126
        self.tmax=186

    def make_plots(self, epoch):
        '''Create a new dataset object, run all the steps, and make all the plots'''
        self.output_filename=os.path.join(self.outputdirectory,'epoch%d'%(epoch))
        self.dataset = submitted.Dataset(epoch)
        prev_error=np.Inf
        print("current epoch is "+str(epoch))
        for n in range(0,len(submitted.steps)):
            getattr(self.dataset, 'set_' + submitted.steps[n])()
            if epoch < 0 or epoch % 20 == 0:
                print('Creating plots for epoch %d step %d (%s)'%(epoch,n,submitted.steps[n]))
                plotter_method = getattr(self, 'plot_%s'%(submitted.steps[n]))
                self.figs[n].clf()
                plotter_method(self.figs[n], self.dataset)
                self.figs[n].savefig(self.output_filename + '_step%d.png'%(n))

        return(self.dataset.label-self.dataset.activation[:,4])

    def plot_model(self, f, dataset):
        '''Plot the model'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.model)
        a.set_title('Model weights, epoch %d'%(dataset.epoch))

    def plot_activation(self, f, dataset):
        '''Plot the activations'''
        aa = f.subplots(nrows=5,ncols=1,sharex=True)
        labels = ['Cell','Input','Forget','Output','Pred']
        for row in (0,1,2,3,4):
            aa[row].plot(dataset.activation[self.tmin:self.tmax,row])
            aa[row].set_ylabel(labels[row])
        aa[0].set_title('Activations')
        aa[4].set_xlabel('Sample index')

    def plot_deriv(self, f, dataset):
        '''Plot the activation function derivatives'''
        aa = f.subplots(nrows=4,ncols=1,sharex=True)
        labels = ['Cell','Input','Forget','Output']
        for row in (0,1,2,3):
            aa[row].plot(dataset.deriv[self.tmin:self.tmax,row])
            aa[row].set_ylabel(labels[row])
        aa[0].set_title('Derivatives')
        aa[3].set_xlabel('Sample index')

    def plot_partial(self, f, dataset):
        '''Plot the prediction, label, and partial'''
        aa = f.subplots(nrows=3,ncols=1,sharex=True)
        aa[0].plot(dataset.activation[self.tmin:self.tmax,4])
        aa[0].set_title('Prediction')
        aa[1].plot(dataset.label[self.tmin:self.tmax])
        aa[1].set_xlabel('Label')
        aa[2].plot(dataset.partial[self.tmin:self.tmax])
        aa[2].set_xlabel('Partial')

    def plot_bptt(self, f, dataset):
        '''Plot the backprop'''
        aa = f.subplots(nrows=5,ncols=1,sharex=True)
        labels = ['Cell','Input','Forget','Output','Pred']
        for row in (0,1,2,3,4):
            aa[row].plot(dataset.bptt[self.tmin:self.tmax,row])
            aa[row].set_ylabel(labels[row])
        aa[0].set_title('Backprop through time')
        aa[4].set_xlabel('Sample index')

    def plot_gradient(self, f, dataset):
        '''Plot the gradient'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.gradient)
        a.set_title('Model gradients, epoch %d'%(dataset.epoch))

    def plot_update(self, f, dataset):
        '''Plot the gradient'''
        a = f.add_subplot(1,1,1)
        a.imshow(dataset.update)
        a.set_title('Updated model, epoch %d'%(dataset.epoch))

def plot_sum_squared_error(epoch,sum_squared_error,output_directory):
    '''Plot sum squared error over time'''
    f=matplotlib.pyplot.figure()
    a=f.add_subplot(1,1,1)
    a.plot(sum_squared_error)
    a.set_title('Sum squared error over epoch')
    f.savefig(output_directory+"/epoch"+str(epoch)+"_sum_squared_error"+".png")

#####################################################
# If this is called from the command line,
# create a plotter with the specified arguments (or with default arguments),
# then create the corresponding plots in the 'make_cool_plots_outputs' directory.
if __name__ == '__main__':
    output_directory='vis'
    plotter=Plotter(output_directory)
    sqe=[]
    total_epochs=141
    for epoch in range(-1,total_epochs):
        error=plotter.make_plots(epoch)
        sqe.append(np.mean(np.square(error)))
    plot_sum_squared_error(total_epochs,sqe,output_directory)

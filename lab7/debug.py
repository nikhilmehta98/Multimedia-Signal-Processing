import submitted
import numpy as np
import importlib,os

def start_debugging(epoch=0, nsteps_to_run=len(submitted.steps)):
    '''(dataset, step, solution, error) = debug.start_debugging(epoch, nsteps_to_run)

    This function reloads your submitted.py file (in case you have changed it).
    It then creates a new Dataset object, using the specified epoch (default:0),
    so that the new Dataset object will have the properties specified by your newest 
    submitted.py file. The resulting dataset object is returned as "dataset".

    Then it runs the first nsteps_to_run steps specified in your submitted.py file.
    The name of the last-run step is returned to you as "step".

    If a solution.npz file is available for your specified epoch, it is loaded as 
    the "solution" object.  The difference between "step" in the reference solution, 
    versus "step" in your submitted.py dataset object, is returned to you as "error".
    '''
    # This line reloads "submitted", in case you've modified your code
    importlib.reload(submitted)

    # This line creates a new dataset object
    dataset=submitted.Dataset(epoch)

    print('finished loading the dataset')
    
    # This line runs the steps you ask for
    nsteps_to_run = max(0, min(len(submitted.steps), nsteps_to_run))
    print('there are %d steps to run'%(nsteps_to_run))
    for n in range(nsteps_to_run):
        print('running step %d: %s'%(n,submitted.steps[n]))
        fun=getattr(dataset,'set_'+submitted.steps[n])
        fun()

    # This line loads the solutions file, if it exists
    solutions_filename = os.path.join('solutions','epoch%d_solution.npz'%(epoch))
    step = submitted.steps[nsteps_to_run-1]
    if not os.path.isfile(solutions_filename):
        error = 'No reference solution available for epoch %d'%(epoch)
        solution = {}
    else:
        solution = np.load(solutions_filename)
        if (step not in solution) or (not hasattr(dataset,step)):
            error = step + ' does not exist in solution, or does not exist in dataset'
        elif hasattr(solution[step],'shape') and hasattr(getattr(dataset,step),'shape'):
            if solution[step].shape != getattr(dataset,step).shape:
                error = 'Solution=%s, Dataset=%s'%(solution[step].shape,getattr(dataset,step).shape)
            else:
                error = getattr(dataset,step)-solution[step]
        elif solution[step] == getattr(dataset,step):
            error='None'
        else:
            error='The objects differ'

    return(dataset, step, solution, error)
    

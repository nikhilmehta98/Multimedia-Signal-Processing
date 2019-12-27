import unittest,json,submitted,score
from gradescope_utils.autograder_utils.decorators import weight
from parameterized import parameterized

# Define the processing steps
steps = [
    'model',
    'activation',
    'deriv',
    'partial',
    'bptt',
    'gradient',
    'update'
]

# Testsequence
class TestSequence(unittest.TestCase):
    @parameterized.expand([
        ['file%d_step%d'%(epoch,stepnum),epoch,stepnum]
        for epoch in (-1,0,50,100)
        for stepnum in range(len(steps))
    ])
    
    @weight(3.571428571428)
    def test_sequence(self, name, epoch, stepnum):
        # Create the dataset object, and load its data
        dataset = submitted.Dataset(epoch)

        # Load the reference solutions
        filename = 'epoch%d'%(epoch)
        with open('solutions/%s.json'%(filename)) as f:
            ref = json.load(f)            

        # Perform all steps prior to the one we want to test ---
        # this wastes a lot of time, but I'm not sure how else to do it,
        # since the dataset is created anew for each test!
        for n in range(stepnum+1):
            step=steps[n]
            getattr(dataset, 'set_' + step)()

        # Get the attribute being tested by this particular unit test
        step = steps[stepnum]
        x = getattr(dataset, step)
        # If this is 'model', then test, only the bottom two rows;
        #   there are more than one possible solutions for the first two rows
        if step=='model' or step=='update':
            x = x[1:4,:]
        # Check that its golden-mean-projections all have the correct values
        self.assertTrue(score.validate_data_content(x,ref[step]['content'],name,stepnum,0.001))

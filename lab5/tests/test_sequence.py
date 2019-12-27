import unittest,json,submitted,score
from gradescope_utils.autograder_utils.decorators import weight
from parameterized import parameterized

# Define the processing steps
steps = [
    'surprisal',
    'alphahat',
    'betahat',
    'gamma',
    'xi',
    'mu',
    'var',
    'tpm',
]

# Testsequence
class TestSequence(unittest.TestCase):
    @parameterized.expand([
        ['file%d_step%d'%(epoch,stepnum),epoch,stepnum]
        for epoch in range(0,2)
        for stepnum in range(len(steps))
    ])
    
    @weight(6.25)
    def test_sequence(self, name, epoch, stepnum):
        # Create the KNN object, and load its data
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
        # If it's not "psi", test it
        x = getattr(dataset, step)
        # Check that its golden-mean-projections all have the correct values
        self.assertTrue(score.validate_data_content(x,ref[step]['content'],name,stepnum,0.001))

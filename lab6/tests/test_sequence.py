import unittest,json,submitted,score
from gradescope_utils.autograder_utils.decorators import weight
from parameterized import parameterized

# Define the processing steps
steps = [
    'layer1validate',
    'layer1test',
    'layer2validate',
    'layer2test',
    'grad2validate',
    'grad1validate',
    'corners',
    'cartesian2barycentric',
    'barycentric',
    'pix2tri',
    'refcoordinate',    
    'intensities'
]

# Testsequence
class TestSequence(unittest.TestCase):
    @parameterized.expand([
        ['file%d_step%d'%(testcase,stepnum),testcase,stepnum]
        for testcase in range(5)
        for stepnum in (0,1,2,3,4,5,6,7,8)
    ])
    
    @weight(2.2222222222222222222222222)
    def test_sequence(self, name, testcase, stepnum):
        # Create the KNN object, and load its data
        dataset = submitted.Dataset(testcase)

        # Load the reference solutions
        filename = 'testcase%d'%(testcase)
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

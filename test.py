import numpy as np
import torch
import unittest

def original_div_term(model_dimension):
    return np.exp(np.arange(0, model_dimension, 2, dtype=np.float32) * -(np.log(10_000.0) / model_dimension))

def your_div_term(model_dimension, wave_length=10_000.0):
    return 1 / (wave_length ** ((torch.arange(0, model_dimension, 2, dtype=torch.float32) / model_dimension)))

class TestPositionalEncodingMethods(unittest.TestCase):

    def test_equivalence_of_methods(self):
        model_dimension = 512
        original = original_div_term(model_dimension)
        yours = your_div_term(model_dimension)
        
        difference = np.linalg.norm(original - yours)
        
        tolerance = 1e-6
        
        self.assertLessEqual(difference, tolerance, "The two methods produce outputs that differ more than the allowed tolerance.")

if __name__ == '__main__':
    unittest.main()
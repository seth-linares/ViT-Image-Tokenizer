import unittest
from image_tokenization import ImageTokenizer  
from PIL import Image
import numpy as np
import os
class TestImageTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.test_dir = r"C:\Users\sethl\Desktop"
        cls.image_dimension = 224
        cls.patch_size = 32

    def test_size_standardizer(self):
        tokenizer = ImageTokenizer(self.test_dir, self.patch_size, self.image_dimension)
        test_image_path = os.path.join(self.test_dir, os.listdir(self.test_dir)[0])  
        img = Image.open(test_image_path)
        standardized_img = tokenizer.size_standardizer(img)
        
        self.assertEqual(standardized_img.size, (self.image_dimension, self.image_dimension), "Standardized image dimensions are incorrect.")

    def test_image_to_patch(self):
        tokenizer = ImageTokenizer(self.test_dir, self.patch_size, self.image_dimension)
        test_image_path = os.path.join(self.test_dir, os.listdir(self.test_dir)[0])  
        img = Image.open(test_image_path)
        standardized_img = tokenizer.size_standardizer(img)
        patches = tokenizer.image_to_patch(standardized_img)
        
        expected_patch_count = (self.image_dimension // self.patch_size) ** 2
        self.assertEqual(len(patches), expected_patch_count, "Incorrect number of patches extracted.")
        
        for patch in patches:
            self.assertEqual(patch.size, (self.patch_size, self.patch_size), "Patch dimensions are incorrect.")

    def test_patch_to_vec(self):
        tokenizer = ImageTokenizer(self.test_dir, self.patch_size, self.image_dimension)
        patch = Image.new("RGB", (self.patch_size, self.patch_size), "red")
        patch_vec = tokenizer.patch_to_vec(patch)
        
        expected_vec_length = self.patch_size * self.patch_size * 3 
        self.assertEqual(len(patch_vec), expected_vec_length, "Vectorized patch length is incorrect.")

if __name__ == "__main__":
    unittest.main()
#%%
import os
from PIL import Image
import numpy as np

#%%

class ImageTokenizer:
    def __init__(self, 
                 directory_path: str, 
                 patch_size: int = 32, 
                 image_dimension: int = 224, 
                 background_color: tuple = (255, 255, 255), 
                 interpolation_method=Image.BILINEAR, 
                 bias=None) -> None:
        
        valid_extensions = (".png", ".jpg", ".jpeg")
        if directory_path is not None and os.path.isdir(directory_path):
            self.image_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.lower().endswith(valid_extensions)]
        else:
            raise Exception("Invalid directory.")
        
        self.patch_size = patch_size
        self.image_dimensions = (image_dimension, image_dimension)
        self.background_color = background_color
        self.bias = bias
        self.interpolation_method = interpolation_method



    def generate_positional_encodings(self, num_patches):
        pos_encodings = np.zeros((num_patches, self.patch_size * self.patch_size * 3)) 
        for pos in range(num_patches):
            for i in range((self.patch_size * self.patch_size * 3)):
                pos_encodings[pos, i] = pos + i / 10_000 ** (2 * (i // 2) / (self.patch_size * self.patch_size * 3))
        return pos_encodings
    

    def process_images(self):
        all_patches_vecs = []

        for path in self.image_paths:
            img = Image.open(path)
            standardized_img = self.size_standardizer(img)
            patches = self.image_to_patch(standardized_img)

            pos_encodings = self.generate_positional_encodings(len(patches))

            for idx, patch in enumerate(patches):
                patch_vec = self.patch_to_vec(patch)
                patch_vec += pos_encodings[idx]
                all_patches_vecs.append(patch_vec)

        return all_patches_vecs


    
    def size_standardizer(self, img):
        change_factor = min(self.image_dimensions[0] / img.size[0], self.image_dimensions[1] / img.size[1])
        new_image_dimension = (int(img.size[0] * change_factor), int(img.size[1] * change_factor))
        img = img.resize(new_image_dimension, self.interpolation_method)


        new_img = Image.new("RGB", self.image_dimensions, self.background_color)
        paste_position = ((self.image_dimensions[0] - img.size[0]) // 2, (self.image_dimensions[1] - img.size[1]) // 2)
        new_img.paste(img, paste_position)

        return new_img

    
    def image_to_patch(self, image: Image):
        patches = []
        
        pad_width = (self.patch_size - (image.size[0] % self.patch_size)) % self.patch_size
        pad_height = (self.patch_size - (image.size[1] % self.patch_size)) % self.patch_size
        
        if pad_width > 0 or pad_height > 0:
            padded_image = Image.new("RGB", (image.size[0] + pad_width, image.size[1] + pad_height), self.background_color)
            padded_image.paste(image, (0, 0))
        else:
            padded_image = image
        
        for i in range(0, padded_image.size[0], self.patch_size):
            for j in range(0, padded_image.size[1], self.patch_size):
                patch = padded_image.crop((i, j, i + self.patch_size, j + self.patch_size))
                patches.append(patch)
        
        return patches



    def patch_to_vec(self, patch, projection_matrix=None):
        flattened_patch = np.array(patch).reshape(-1)
        
        if projection_matrix is not None:
            assert projection_matrix.shape[1] == flattened_patch.shape[0], "Projection matrix and patch vector size mismatch."
            projected_patch = np.dot(projection_matrix, flattened_patch)
            
            if self.bias is not None:
                projected_patch += self.bias
                
            return projected_patch
        else:
            return flattened_patch

#%%
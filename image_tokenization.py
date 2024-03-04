#%%
import os
from PIL import Image
import numpy as np

#%%

class ImageTokenizer:
    def __init__(self, 
                 directory_path: str, 
                 patch_size: int = 32, 
                 model_dimension = 512,
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
        self.model_dimension = model_dimension # hyper parameter based on the dimension of the model you choose
        self.image_dimensions = (image_dimension, image_dimension)
        self.background_color = background_color
        self.bias = bias
        self.interpolation_method = interpolation_method

        self.num_patches = (self.image_dimensions[0] // self.patch_size) ** 2 

        # Calculate the flattened patch vector size
        self.patch_vector_size = (patch_size ** 2) * 3  # 3 is for the RGB channels

        # Initialize the projection matrix and bias
        self.projection_matrix = np.random.randn(self.patch_vector_size, model_dimension).astype(np.float32)
        self.bias = np.random.randn(model_dimension).astype(np.float32)
        

        # Typically the class token is a learnable parameter, but for simplicity and to keep our code "framework agnostic", we'll initialize it as a zero vector.
        self.class_token_vector = np.zeros((1, model_dimension))

    # d-dimensional vector that contains information on the position of the patch in the image.
    def generate_positional_encodings(self):

        # Each row is a patch and each column is a dimension.
        positional_encodings = np.zeros((self.num_patches, self.model_dimension))

        # We take each row from 0 -> num_patches - 1 and then make it into a column vector so that it can be added to the patch embedding via broadcasting.
        position = np.arange(0, self.num_patches, dtype=np.float32)[:, np.newaxis]

        # Controls frequency scaling across dimensions for smooth positional encoding
        div_term = np.exp(np.arange(0, self.model_dimension, 2, dtype=np.float32) * -(np.log(10_000.0) / self.model_dimension))

        # The even indices of the positional encodings are updated with the sin of the position multiplied by the div_term.
        positional_encodings[:, 0::2] = np.sin(position * div_term)

        # Same as above but for the odd indices.
        positional_encodings[:, 1::2] = np.cos(position * div_term)
        return positional_encodings

    

    def process_images(self):
        all_image_encodings = []

        for path in self.image_paths:
            img_patches_vecs = []
            img = Image.open(path)
            standardized_img = self.size_standardizer(img)
            patches = self.image_to_patch(standardized_img)

            pos_encodings = self.generate_positional_encodings()

            # For each patch, we flatten it and then project it into the model's dimensionality. We then add the positional encoding to the projected patch vector.
            for idx, patch in enumerate(patches):
                # Flatten the patch
                patch_vec = self.patch_to_vec(patch)  
                # Add positional encoding to the patch vector
                patch_vec += pos_encodings[idx]
                # Append the patch vector to the list of patch vectors
                img_patches_vecs.append(patch_vec)

            # Attach the fixed class token to the beginning of the sequence of patch vectors as this will serve as a special "first token" that will be used to aggregate information from the patches. Here we are using a simple zero vector as the class token, but in practice you would use TensorFlow/PyTorch to create a learnable parameter. However, for simplicity and to keep our code framework agnostic, it's just a zero vector.
            image_encoding = np.vstack([self.class_token_vector, img_patches_vecs])
            
            # Append the image encoding to the list of all image encodings
            all_image_encodings.append(image_encoding)

        return all_image_encodings


    
    def size_standardizer(self, img):
        # Resizes the image to the specified dimensions while maintaining the aspect ratio
        change_factor = min(self.image_dimensions[0] / img.size[0], self.image_dimensions[1] / img.size[1]) # Takes larger dimension and scales it down to meet the target dimension
        new_image_dimension = (int(img.size[0] * change_factor), int(img.size[1] * change_factor))
        img = img.resize(new_image_dimension, self.interpolation_method)


        new_img = Image.new("RGB", self.image_dimensions, self.background_color)
        paste_position = ((self.image_dimensions[0] - img.size[0]) // 2, (self.image_dimensions[1] - img.size[1]) // 2)
        new_img.paste(img, paste_position)

        return new_img

    
    def image_to_patch(self, image: Image):
        # List to collect the patches we make from our image
        patches = []
        # Ensuring that the image is padded to allow for even division into patches
        pad_width = (self.patch_size - (image.size[0] % self.patch_size)) % self.patch_size
        pad_height = (self.patch_size - (image.size[1] % self.patch_size)) % self.patch_size
        
        # If the image is not already the correct size, we pad it with the background color
        if pad_width > 0 or pad_height > 0:
            padded_image = Image.new("RGB", (image.size[0] + pad_width, image.size[1] + pad_height), self.background_color)
            padded_image.paste(image, (0, 0))
        else:
            padded_image = image
        
        # Goes through the image in patch_size increments and crops out the patch
        for i in range(0, padded_image.size[0], self.patch_size):
            for j in range(0, padded_image.size[1], self.patch_size):
                patch = padded_image.crop((i, j, i + self.patch_size, j + self.patch_size))
                patches.append(patch)
        
        return patches



    def patch_to_vec(self, patch):
        # Flatten the patch
        flattened_patch = np.array(patch).reshape(-1)  
        # Apply projection and bias
        projected_patch = np.dot(flattened_patch, self.projection_matrix) + self.bias  
        return projected_patch


#%%
import matplotlib.pyplot as plt
import numpy as np

model_dimension = 16
div_term = np.exp(np.arange(0, model_dimension, 2, dtype=np.float32) * -(np.log(10000.0) / model_dimension))

plt.plot(div_term)
plt.xlabel("Dimension")
plt.ylabel("Scaling Factor")
plt.title("Visualization of div_term")
plt.show()
# %%

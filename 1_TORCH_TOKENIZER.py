#%%
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.init as init
from PIL import UnidentifiedImageError

#%%

class ImageTokenizer(Dataset):
    def __init__(self, root_dir, patch_size=32, model_dimension=512, image_dimension=224,interpolation_method=Image.LANCZOS, wave_length=10000, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__() 

        # Code to ensure that the patch and image dimensions are compatible for a clean division of the image
        rem = image_dimension % patch_size
        if image_dimension < patch_size:
            self.image_dimensions = (patch_size, patch_size)
        elif rem != 0:
            self.image_dimensions = (image_dimension - rem, image_dimension - rem)

        self.patch_size = patch_size
        self.model_dimension = model_dimension
        self.num_patches = (image_dimension // patch_size) ** 2

        

        self.wave_length = wave_length
        self.positional_encodings = self.generate_positional_encodings()

        # Torch component initiaization below this point:


        self.dataset = datasets.ImageFolder(root=root_dir, 
                                            transform=transforms.Compose([
                                                transforms.Resize(self.image_dimensions, interpolation_method),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)  # Use imagenet's mean and std by default for normalization
                                            ]))
        self.projection_matrix = torch.nn.Parameter(torch.empty((patch_size**2) * 3, model_dimension))
        self.class_token_vector = torch.nn.Parameter(torch.empty(1, model_dimension))
        self.bias = torch.nn.Parameter(torch.zeros(model_dimension))

        # Apply Xavier uniform initialization
        init.xavier_uniform_(self.projection_matrix)
        init.xavier_uniform_(self.class_token_vector)


        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            image, label = self.dataset[idx]
        except UnidentifiedImageError:
            raise Exception(f"Corrupt image detected at index {idx}!!")
        
        except Exception as e:
            raise Exception(f"Unexpected error with image at index {idx}: {e}")
        
        patches = self.image_to_patches(image)
        projected_patches = self.project_patches(patches) 
        patches_with_token = torch.cat([self.class_token_vector.unsqueeze(0), projected_patches], dim=0)
        patches_with_token += self.positional_encodings.unsqueeze(0) 
        return patches_with_token, label
        
    def generate_positional_encodings(self):
        positional_encodings = torch.zeros((1 + self.num_patches, self.model_dimension))
        position = torch.arange(0, self.num_patches + 1, dtype=torch.float32).unsqueeze(1)
        div_term = 1 / (self.wave_length ** (torch.arange(0, self.model_dimension, 2, dtype=torch.float32) / self.model_dimension))
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        return positional_encodings

    def image_to_patches(self, img_tensor: torch.Tensor):
        """
        Since we have already transformed our image to a tensor in our init method, we can now directly use unfold to get our patches. 
        The great thing about making our image into a tensor before getting to the patching process is that we can now use PyTorch's unfold method to get our patches.
        Unfold lets us extract "sliding local blocks" from the tensor. These blocks represent the patches we want to extract from the image. 

        Normally the looping process would look something like this:
        for i in range(0, image.size[0], patch_size):
            for j in range(0, image.size[1], patch_size):
                patch = padded_image.crop((i, j, i + patch_size, j + patch_size))
                patches.append(patch)
        
        But unfolding allows us to do this in a more efficient way as it uses tensor operations to extract the patches which is preferable over normal Python constructs 
        whenever possible. The reason we have to do 2 unfold operations however is that we need to slide over both the image's width and height to get all the patches. Otherwise,
        the slices would go over the entire height or width of the image, so we wouldn't get square patches.



        Once we have the patches, we now need to flatten them. This is because we need to project them into the model's dimension which will be done in the next method. 
        
        * It is important to note that we need to use contiguous() before we can use view() to flatten the patches. This is because operations like unfold() don't actually change the actual data in memory but rather modify the way we index the original tensor. This can end up causing issues where the logical layout for our tensor is different from the physical layout. This is why we need to call contiguous() to ensure that the tensor is stored in a contiguous block of memory before we can flatten it.
        """
        patches = img_tensor.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, self.patch_size * self.patch_size * 3) 
        return patches

    def project_patches(self, patches):
        """
        We are taking our "out of shape" patches that have been flattened and then projecting them into the model's dimension 
        so that we can use them in other operations which require the model's dimension to be consistent (i.e. the positional encodings).

        The projection matrix is a learnable parameter that we initialize using Xavier uniform initialization. The formula for applying linear projection is:
        Y = XW + b
        where Y is the output, X is the input, W is the projection matrix and b is the bias.
        """
        return torch.matmul(patches, self.projection_matrix) + self.bias
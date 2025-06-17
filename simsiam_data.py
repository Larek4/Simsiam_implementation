import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image # For reading PNG images
import os

# --- Custom Solarization Transform ---
# This class acts as a wrapper for torchvision's solarize function.
# It expects the input image to be a PIL Image or a Tensor in the 0-255 range.
class Solarization(object):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        # torchvision.functional.solarize expects a PIL Image or a Tensor,
        # and operates on pixel values directly.
        # It's important that this transform is applied BEFORE ToTensor()
        # if the threshold is in the 0-255 range, which it is.
        return transforms.functional.solarize(img, self.threshold)

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, transform=None, transform_v2=None):
        """
        Args:
            image_dir (str): Directory with all the image files.
            transform (callable, optional): Optional transform to be applied on view 1.
            transform_v2 (callable, optional): Optional transform to be applied on view 2.
        """
        self.image_dir = image_dir
        # List all PNG files in the directory.
        # Ensure the path is correct relative to where you run your script, or absolute.
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith('.png') # Assuming PNGs, add other formats if necessary
        ]
        self.transform = transform # Transform pipeline for the first view
        self.transform_v2 = transform_v2 # Transform pipeline for the second view
        
        if not self.image_paths:
            print(f"Warning: No PNG images found in {image_dir}. Please check the path and ensure images are generated.")
            print(f"Current working directory: {os.getcwd()}") # Helpful for debugging paths


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load the image using PIL (Pillow)
        # Convert to RGB to ensure 3 channels, even if source is grayscale/RGBA (though your PNGs are RGB)
        raw_image = Image.open(img_path).convert('RGB')

        # Apply the augmentations to get two different views
        # Each view gets its specific pipeline (v1 and v2)
        if self.transform and self.transform_v2:
            view1 = self.transform(raw_image)
            view2 = self.transform_v2(raw_image)
        else:
            # Fallback for testing or if no transforms are provided
            view1 = transforms.ToTensor()(raw_image)
            view2 = transforms.ToTensor()(raw_image)
        
        return view1, view2

# --- Data Augmentation Pipelines for SimSiam ---
# These pipelines apply a series of random transformations to the images.
# The success of SimSiam heavily relies on diverse and strong augmentations.

# Define a base set of transformations that operate on PIL Images FIRST
# These transforms are applied to both views before ToTensor().
pil_image_transforms = [
    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)), # Resize and random crop
    transforms.RandomHorizontalFlip(p=0.5), # Random horizontal flip
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, # Color distortion
                               saturation=0.2, hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2), # <--- ADDED: RandomGrayscale (20% chance)
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)) # Gaussian blur
    ], p=0.5),
]

# Define transforms that operate on PyTorch Tensors LAST
# These are applied after ToTensor() (implicitly, as ToTensor() is usually the last PIL->Tensor step).
tensor_transforms_final = [
    transforms.ToTensor(), # Converts PIL Image to PyTorch Tensor (scales to [0,1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
]

# Transform for the first view (v1)
# This includes all PIL-based common augmentations, followed by ToTensor and Normalize.
simsiam_transform_v1 = transforms.Compose(pil_image_transforms + tensor_transforms_final)

# Transform for the second view (v2)
# This includes all PIL-based common augmentations PLUS Solarization, followed by ToTensor and Normalize.
# Solarization must happen BEFORE ToTensor() as it operates on 0-255 pixel values.
simsiam_transform_v2 = transforms.Compose(pil_image_transforms + [
    transforms.RandomApply([Solarization()], p=0.2) # <--- ADDED: Solarization (20% chance)
] + tensor_transforms_final)


# --- How to test your dataset (optional, but highly recommended for verification) ---
# This block runs only when simsiam_data.py is executed directly.
if __name__ == "__main__":
    # IMPORTANT: Ensure this path points to your actual PNG images.
    # If your `barlow_data` folder is directly inside 'seminar', this path is correct.
    image_folder = 'barlow_data/train' # This is where your PNGs are saved.

    # Create a dummy folder and image for testing if it doesn't exist
    if not os.path.exists(image_folder):
        print(f"Creating dummy directory for testing: {image_folder}")
        os.makedirs(image_folder)
        try:
            # Create a dummy 3-channel PNG image for demonstration
            dummy_img = Image.new('RGB', (256, 256), color = 'red')
            dummy_img.save(os.path.join(image_folder, 'dummy_satellite_001.png'))
            print(f"Created a dummy image for testing in {image_folder}")
        except Exception as e:
            print(f"Could not create dummy image (check permissions/disk space): {e}")

    print(f"\nAttempting to load dataset from: {image_folder}")
    # Pass both transform pipelines to the dataset instance
    dataset = SatelliteDataset(image_folder, transform=simsiam_transform_v1, transform_v2=simsiam_transform_v2)

    if len(dataset) > 0:
        print(f"Successfully loaded {len(dataset)} images.")
        try:
            view1, view2 = dataset[0] # Get the first pair of augmented views
            print(f"Shape of View 1: {view1.shape}") # Should be [C, H, W] e.g., [3, 224, 224]
            print(f"Shape of View 2: {view2.shape}") # Should be [3, 224, 224]
            print(f"Data type: {view1.dtype}")    # Should be torch.float32
            
            # Verify values are normalized (e.g., typically between -2 and 2 after normalization)
            print(f"Min value of View 1: {view1.min().item():.4f}")
            print(f"Max value of View 1: {view1.max().item():.4f}")
            print(f"Min value of View 2: {view2.min().item():.4f}")
            print(f"Max value of View 2: {view2.max().item():.4f}")

        except Exception as e:
            print(f"Error accessing dataset item (check if images are valid): {e}")
            print("This might happen if your dataset is empty or images are corrupted.")
    else:
        print("Dataset is empty. Please ensure your conversion script has run successfully and saved images to the specified 'image_folder'.")
        print(f"Expected image folder: {os.path.abspath(image_folder)}")
        print(f"Files found in this folder: {os.listdir(image_folder) if os.path.exists(image_folder) else 'Folder does not exist.'}")
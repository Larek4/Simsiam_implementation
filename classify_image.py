import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import os
import json

# Import your model structures from their respective files
from simsiam_model import SimSiam
from finetune_model import FineTuneModel # Re-using FineTuneModel class for inference

# --- Configuration ---
# Path to your BEST fine-tuned model checkpoint
MODEL_CHECKPOINT_PATH = 'finetuned_eurosat_best_model.pth' 

# Path to the EuroSAT label map JSON file
LABEL_MAP_PATH = 'datae/EuroSAT/label_map.json' 

# Path to the image you want to classify (CHANGE THIS!)
IMAGE_TO_CLASSIFY_PATH = 'datae/EuroSAT/Pasture/Pasture_30.jpg' # Example path, replace with your test PNG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def classify_single_image(image_path, model_path, label_map_path):
    """
    Classifies a single image using the fine-tuned model.

    Args:
        image_path (str): Path to the input image (PNG or JPG).
        model_path (str): Path to the saved .pth model checkpoint.
        label_map_path (str): Path to the label_map.json file.
    """
    # 1. Load the label map
    try:
        with open(label_map_path, 'r') as f:
            class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"Loaded class mapping: {class_to_idx}")
    except FileNotFoundError:
        print(f"Error: label_map.json not found at {label_map_path}. Please check the path.")
        return

    # 2. Load the model
    print(f"Loading model from {model_path}...")
    try:
        # Reconstruct the base_encoder (ResNet-50) architecture
        base_encoder_for_loading = models.resnet50(weights=None)
        # Create a dummy SimSiam model to correctly load the backbone's state
        simsiam_full_model_dummy = SimSiam(base_encoder_for_loading)
        
        # Create the fine-tuning model structure
        # We need the NUM_CLASSES to initialize FineTuneModel, which is 10 for EuroSAT
        # We also need the backbone which comes from the dummy SimSiam model
        model = FineTuneModel(simsiam_full_model_dummy.backbone, num_classes=len(class_to_idx)) 
        
        # Load the state_dict from the saved checkpoint
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully and set to evaluation mode.")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}. Please check the path and ensure pre-training and fine-tuning are complete.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load and preprocess the image
    print(f"Loading image from {image_path}...")
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Apply the same transformations used during fine-tuning
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

    # 4. Move input to the same device as the model
    input_batch = input_batch.to(DEVICE)

    # 5. Make prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(input_batch)

    # Get probabilities (optional, but good for understanding confidence)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get the predicted class index
    predicted_idx = torch.argmax(probabilities).item()
    
    # Convert index to class name
    predicted_class = idx_to_class[predicted_idx]

    # 6. Print results
    print("\n--- Classification Result ---")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class Index: {predicted_idx}")
    print(f"Predicted Class Name: {predicted_class}")
    print(f"Confidence: {probabilities[predicted_idx].item():.4f}")
    print("----------------------------")

if __name__ == "__main__":
    # Ensure this path points to a PNG or JPG image you want to test!
    # You can pick one from your EuroSAT dataset (e.g., in data/EuroSAT/River/)
    # or use any other satellite image in RGB format.
    classify_single_image(IMAGE_TO_CLASSIFY_PATH, MODEL_CHECKPOINT_PATH, LABEL_MAP_PATH)

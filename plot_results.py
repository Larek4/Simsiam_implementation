import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import json
from PIL import Image
from torchvision import transforms
import torchvision.models as models 

# Import model structures for example classification visualization
from simsiam_model import SimSiam
from finetune_model import FineTuneModel 

# --- Configuration for Plotting and Examples ---
LINEAR_EVAL_METRICS_PATH = 'linear_eval_metrics.pth'
FINETUNE_METRICS_PATH = 'finetune_metrics.pth'
PRETRAIN_METRICS_PATH = 'pretrain_metrics.pth'

# For example classifications:
FINETUNED_MODEL_PATH = 'finetuned_eurosat_best_model.pth' 
LABEL_MAP_PATH = 'datae/EuroSAT/label_map.json'

# Paths to example images for qualitative visualization.
# Adjust these to real image paths from your EuroSAT dataset, including some misclassified ones if you know them.
# Example: Pick some images from your datae/EuroSAT/Test/ (or Validation/) set.
EURO_SAT_ROOT = 'datae/EuroSAT'
EXAMPLE_IMAGE_PATHS_AND_TRUE_LABELS = [
    (os.path.join(EURO_SAT_ROOT, 'River/River_1000.jpg'), 'River'),
    (os.path.join(EURO_SAT_ROOT, 'Forest/Forest_1001.jpg'), 'Forest'),
    (os.path.join(EURO_SAT_ROOT, 'AnnualCrop/AnnualCrop_100.jpg'), 'AnnualCrop'),
    (os.path.join(EURO_SAT_ROOT, 'Industrial/Industrial_200.jpg'), 'Industrial'),
    # Add more paths and their true labels here (e.g., a few correct, a few incorrect if you find them)
]

# For augmentation visualization
IMAGE_PATH_FOR_AUG_VIZ = os.path.join(EURO_SAT_ROOT, 'Residential/Residential_1.jpg') # Pick a sample image from your training data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use CPU for plotting if GPU is not available

# --- Plotting Functions ---

def plot_pretraining_metrics(metrics_file, save_dir='plots'):
    """Plots pre-training loss and learning rate schedule."""
    metrics = torch.load(metrics_file)
    epochs = range(1, len(metrics['epoch_losses']) + 1)

    os.makedirs(save_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['epoch_losses'], label='Average Loss')
    plt.title('SimSiam Pre-training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Cosine Similarity)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pretrain_loss_curve.png'))
    plt.show()

    # Plot Learning Rate Schedule
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['epoch_learning_rates'], label='Learning Rate')
    plt.title('SimSiam Pre-training Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pretrain_lr_schedule.png'))
    plt.show()

    # --- FIX: Conditionally format total_training_time_seconds ---
    total_time = metrics.get('total_training_time_seconds')
    if total_time is not None:
        print(f"Pre-training total time: {total_time:.2f} seconds")
    else:
        print("Pre-training total time: N/A (run may have been interrupted)")
    # --- END FIX ---


def plot_evaluation_metrics(metrics_file, title_prefix, save_dir='plots'):
    """Plots training and validation loss/accuracy curves for evaluation phases."""
    metrics = torch.load(metrics_file)
    epochs = range(1, len(metrics['train_losses']) + 1)

    os.makedirs(save_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_losses'], label='Train Loss')
    plt.plot(epochs, metrics['val_losses'], label='Validation Loss')
    plt.title(f'{title_prefix} - Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in metrics['train_accuracies']], label='Train Accuracy')
    plt.plot(epochs, [acc * 100 for acc in metrics['val_accuracies']], label='Validation Accuracy')
    plt.title(f'{title_prefix} - Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100) # Accuracy typically from 0 to 100

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title_prefix.lower().replace(" ", "_")}_metrics_plot.png'))
    plt.show()

    # --- FIX: Conditionally format total_training_time_seconds ---
    total_time = metrics.get('total_training_time_seconds')
    if total_time is not None:
        print(f"{title_prefix} total training time: {total_time:.2f} seconds")
    else:
        print(f"{title_prefix} total training time: N/A (run may have been interrupted)")
    # --- END FIX ---
    print(f"{title_prefix} best validation accuracy: {metrics.get('best_val_accuracy', 'N/A')*100:.2f}%")
    print(f"{title_prefix} final test accuracy: {metrics.get('final_test_accuracy', 'N/A')*100:.2f}%")


def plot_test_accuracy_comparison(linear_eval_metrics_file, finetune_metrics_file, save_dir='plots'):
    """Compares final test accuracies of linear evaluation and full fine-tuning."""
    linear_metrics = torch.load(linear_eval_metrics_file)
    finetune_metrics = torch.load(finetune_metrics_file)

    linear_acc = linear_metrics.get('final_test_accuracy', 0.0) * 100
    finetune_acc = finetune_metrics.get('final_test_accuracy', 0.0) * 100

    labels = ['Linear Evaluation', 'Full Fine-tuning']
    accuracies = [linear_acc, finetune_acc]

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accuracies, color=['skyblue', 'lightcoral'])
    plt.ylim(min(accuracies) * 0.95 if min(accuracies) > 0 else 0, 100) # Adjust y-axis
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_accuracy_comparison.png'))
    plt.show()


def display_classification_examples(image_paths_and_true_labels, model_path, label_map_path, save_dir='plots'):
    """
    Displays example image classifications (true vs. predicted).
    Requires the fine-tuned model and label map.
    """
    # 1. Load the label map
    try:
        with open(label_map_path, 'r') as f:
            class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
    except FileNotFoundError:
        print(f"Error: label_map.json not found at {label_map_path}. Cannot display examples.")
        return

    # 2. Load the model
    print(f"\nLoading fine-tuned model from {model_path} for example display...")
    try:
        base_encoder_for_loading = models.resnet50(weights=None)
        simsiam_full_model_dummy = SimSiam(base_encoder_for_loading)
        model = FineTuneModel(simsiam_full_model_dummy.backbone, num_classes=len(class_to_idx)) 
        
        # Load state_dict, map to CPU if GPU is not needed/available for plotting
        model.load_state_dict(torch.load(model_path, map_location='cpu')) 
        model.to('cpu') # Ensure model is on CPU for display functions to avoid GPU memory issues during plotting
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully for example display.")
    except FileNotFoundError:
        print(f"Error: Fine-tuned model checkpoint not found at {model_path}.")
        return
    except Exception as e:
        print(f"Error loading model for example display: {e}")
        return
    
    # 3. Define preprocessing transform (same as used during fine-tuning)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Create plot
    num_examples = len(image_paths_and_true_labels)
    plt.figure(figsize=(10, 3 * num_examples)) # Adjust figure size as needed

    os.makedirs(save_dir, exist_ok=True)

    for i, (image_path, true_label_str) in enumerate(image_paths_and_true_labels):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to('cpu') # Ensure input is on CPU for model if model is on CPU

            with torch.no_grad():
                output = model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = idx_to_class[predicted_idx]
            confidence = probabilities[predicted_idx].item()

            # Reverse normalization for display (to get 0-1 pixel values)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            display_image = (input_tensor[0] * std + mean).permute(1, 2, 0).numpy()
            display_image = np.clip(display_image, 0, 1) # Clip values to [0,1] range

            plt.subplot(num_examples, 1, i + 1)
            plt.imshow(display_image)
            title_color = 'green' if predicted_class == true_label_str else 'red'
            plt.title(f"True: {true_label_str} | Predicted: {predicted_class} ({confidence:.2f})", color=title_color)
            plt.axis('off')

        except Exception as e:
            print(f"Could not process image {image_path}: {e}")
            continue

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'classification_examples.png'))
    plt.show()


def visualize_augmentation_pipeline(image_path, num_examples=2, save_dir='plots'):
    """
    Visualizes the data augmentation pipeline (original vs. two augmented views).
    Requires defining the transforms (copied from simsiam_data.py).
    """
    # Redefine transforms to ensure they are available here for visualization
    class Solarization(object):
        def __init__(self, threshold=128):
            self.threshold = threshold
        def __call__(self, img):
            return transforms.functional.solarize(img, self.threshold)

    pil_image_transforms = [
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
    ]
    tensor_transforms_final = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    simsiam_transform_v1 = transforms.Compose(pil_image_transforms + tensor_transforms_final)
    simsiam_transform_v2 = transforms.Compose(pil_image_transforms + [transforms.RandomApply([Solarization()], p=0.2)] + tensor_transforms_final)

    original_image = Image.open(image_path).convert('RGB')

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5 * num_examples))

    plt.subplot(num_examples, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    for i in range(num_examples):
        # Apply transforms and convert back to numpy for plotting
        view1_tensor = simsiam_transform_v1(original_image)
        view2_tensor = simsiam_transform_v2(original_image)

        # Reverse normalization for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        view1_display = (view1_tensor * std + mean).permute(1, 2, 0).numpy()
        view2_display = (view2_tensor * std + mean).permute(1, 2, 0).numpy()

        # Clip values to [0,1] range for display
        view1_display = np.clip(view1_display, 0, 1)
        view2_display = np.clip(view2_display, 0, 1)


        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(view1_display)
        plt.title(f"View 1 (Augmented {i+1})")
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(view2_display)
        plt.title(f"View 2 (Augmented {i+1})")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'augmentation_examples.png'))
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure 'plots' directory exists
    os.makedirs('plots', exist_ok=True)

    # 1. Plot Pre-training Metrics
    print(f"Plotting pre-training metrics from {PRETRAIN_METRICS_PATH}...")
    plot_pretraining_metrics(PRETRAIN_METRICS_PATH)

    # 2. Plot Linear Evaluation Metrics
    print(f"Plotting linear evaluation metrics from {LINEAR_EVAL_METRICS_PATH}...")
    plot_evaluation_metrics(LINEAR_EVAL_METRICS_PATH, "Linear Evaluation")

    # 3. Plot Fine-tuning Metrics
    print(f"Plotting fine-tuning metrics from {FINETUNE_METRICS_PATH}...")
    plot_evaluation_metrics(FINETUNE_METRICS_PATH, "Full Fine-tuning")

    # 4. Plot Test Accuracy Comparison
    print(f"Plotting test accuracy comparison from {LINEAR_EVAL_METRICS_PATH} and {FINETUNE_METRICS_PATH}...")
    plot_test_accuracy_comparison(LINEAR_EVAL_METRICS_PATH, FINETUNE_METRICS_PATH)

    # 5. Display Example Classifications
    print(f"Displaying example classifications from {FINETUNED_MODEL_PATH}...")
    display_classification_examples(EXAMPLE_IMAGE_PATHS_AND_TRUE_LABELS, FINETUNED_MODEL_PATH, LABEL_MAP_PATH)

    # 6. Visualize Augmentation Pipeline
    print(f"Visualizing augmentation pipeline for an example image: {IMAGE_PATH_FOR_AUG_VIZ}...")
    visualize_augmentation_pipeline(IMAGE_PATH_FOR_AUG_VIZ)

    print("\nAll plots generated and saved to the 'plots' directory.")
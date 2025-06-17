# --- Linear_evaluation.py ---

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json

# Import your SimSiam model structure
from simsiam_model import SimSiam

# --- Configuration for Linear Evaluation ---
# IMPORTANT: Ensure this path points to your final pre-trained ResNet-50 checkpoint.
# For example, if you trained for 200 epochs, it should be simsiam_resnet50_epoch_200.pth
PRETRAINED_CHECKPOINT_PATH = 'simsiam_resnet50_epoch_200.pth' 

NUM_CLASSES = 10 # EuroSAT has 10 land use/land cover classes
LINEAR_BATCH_SIZE = 64 
LINEAR_LEARNING_RATE = 0.01 
NUM_LINEAR_EPOCHS = 100 
    
# IMPORTANT: Set this to the actual path to your EuroSAT folder.
# This folder should contain the class subdirectories AND the train.csv, validation.csv, test.csv, label_map.json.
LINEAR_EVAL_IMAGE_ROOT_DIR = 'datae/EuroSAT' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


# --- 1. Custom Dataset for Labeled Data (Updated to read from CSV and JSON) ---
class LabeledSatelliteDataset(Dataset):
    def __init__(self, root_dir, csv_file, label_map_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        try:
            with open(label_map_path, 'r') as f:
                self.class_to_idx = json.load(f)
        except FileNotFoundError:
            print(f"Error: label_map.json not found at {label_map_path}. Please ensure it's there.")
            raise

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}. Please ensure it's there.")
            raise

        if 'Filename' not in self.df.columns or 'ClassName' not in self.df.columns:
            raise ValueError(f"CSV file '{csv_file}' must contain 'Filename' and 'ClassName' columns.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_relative_path = self.df.iloc[idx]['Filename'] 
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        label_name = self.df.iloc[idx]['ClassName'] 
        label = self.class_to_idx[label_name] 
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Skipping this image.")
            raise

        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. Build the Linear Evaluation Model (No changes here) ---
class LinearClassifier(nn.Module):
    def __init__(self, pretrained_backbone, num_classes):
        super().__init__()
        self.backbone = pretrained_backbone
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(2048, num_classes) 
        
    def forward(self, x):
        with torch.no_grad(): 
            features = self.backbone(x)
            features = features.flatten(1) 
        output = self.classifier(features)
        return output

# --- 3. Linear Evaluation Pipeline (Uses explicit Train/Val/Test Splits) ---
def perform_linear_evaluation():
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Loading pre-trained SimSiam backbone...")
    
    base_encoder_for_loading = models.resnet50(weights=None)
    simsiam_full_model = SimSiam(base_encoder_for_loading)
    checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=DEVICE)
    simsiam_full_model.load_state_dict(checkpoint['model_state_dict'])
    pretrained_backbone = simsiam_full_model.backbone
    print("Pre-trained backbone loaded successfully!")

    model = LinearClassifier(pretrained_backbone, NUM_CLASSES)
    model.to(DEVICE)
    print("Linear evaluation model created:")
    print(model)
    
    # --- Data Preparation for EuroSAT (using provided CSV and JSON files) ---
    print("\nInitializing labeled datasets for linear evaluation using CSV splits...")
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.425]) 
    ])

    label_map_path = os.path.join(LINEAR_EVAL_IMAGE_ROOT_DIR, 'label_map.json')
    train_csv_path = os.path.join(LINEAR_EVAL_IMAGE_ROOT_DIR, 'train.csv')
    val_csv_path = os.path.join(LINEAR_EVAL_IMAGE_ROOT_DIR, 'validation.csv')
    test_csv_path = os.path.join(LINEAR_EVAL_IMAGE_ROOT_DIR, 'test.csv')

    train_dataset = LabeledSatelliteDataset(LINEAR_EVAL_IMAGE_ROOT_DIR, train_csv_path, label_map_path, transform=eval_transform)
    val_dataset = LabeledSatelliteDataset(LINEAR_EVAL_IMAGE_ROOT_DIR, val_csv_path, label_map_path, transform=eval_transform)
    test_dataset = LabeledSatelliteDataset(LINEAR_EVAL_IMAGE_ROOT_DIR, test_csv_path, label_map_path, transform=eval_transform) 

    train_dataloader = DataLoader(train_dataset, batch_size=LINEAR_BATCH_SIZE, shuffle=True, 
                                  num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=LINEAR_BATCH_SIZE, shuffle=False, 
                                num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=LINEAR_BATCH_SIZE, shuffle=False, 
                                 num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0, pin_memory=True)

    print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}, Test images: {len(test_dataset)}")
    print(f"Class to index mapping from label_map.json: {train_dataset.class_to_idx}") 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LINEAR_LEARNING_RATE)
    
    print(f"\nStarting linear evaluation training for {NUM_LINEAR_EPOCHS} epochs...")
    best_val_accuracy = 0.0 

    for epoch in range(NUM_LINEAR_EPOCHS):
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar_train = tqdm(train_dataloader, desc=f"Linear Train Epoch {epoch+1}/{NUM_LINEAR_EPOCHS}")
        for images, labels in pbar_train:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad() 
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}", 'TrainAcc': f"{(correct_train/total_train)*100:.2f}%"})

        avg_train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train
        
        model.eval() 
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad(): 
            pbar_val = tqdm(val_dataloader, desc=f"Linear Val Epoch {epoch+1}/{NUM_LINEAR_EPOCHS}")
            for images, labels in pbar_val:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                pbar_val.set_postfix({'Loss': f"{loss.item():.4f}", 'ValAcc': f"{(correct_val/total_val)*100:.2f}%"})
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy*100:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy*100:.2f}%")
        
        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'linear_eval_eurosat_best_model.pth')
            print(f"New best linear evaluation model saved with accuracy: {best_val_accuracy*100:.2f}%")

    print("\nLinear evaluation training complete!")
    print(f"Best validation accuracy achieved during training: {best_val_accuracy*100:.2f}%")
    torch.save(model.state_dict(), 'linear_eval_eurosat_final_model.pth')
    print("Final linear evaluation model saved as 'linear_eval_eurosat_final_model.pth'")

    # Save all collected metrics (before final test accuracy is known)
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy, 
        'final_test_accuracy': None 
    }
    torch.save(metrics, 'linear_eval_metrics.pth')
    print("Linear evaluation metrics saved to linear_eval_metrics.pth")

    # --- FINAL TEST PHASE ---
    print("\n--- Starting final evaluation on the test set ---")
    try:
        model.load_state_dict(torch.load('linear_eval_eurosat_best_model.pth', map_location=DEVICE))
        print("Loaded best model (based on validation accuracy) for final testing.")
    except FileNotFoundError:
        print("Best model checkpoint not found, testing with the final trained model instead.")

    model.eval() 
    correct_test = 0
    total_test = 0
    test_loss = 0.0

    with torch.no_grad():
        pbar_test = tqdm(test_dataloader, desc="Final Test")
        for images, labels in pbar_test:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            pbar_test.set_postfix({'Loss': f"{loss.item():.4f}", 'TestAcc': f"{(correct_test/total_test)*100:.2f}%"})

    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = correct_test / total_test
    print(f"\n--- Final Test Results ---")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")
    print("--------------------------")

    # Update the saved metrics with final test accuracy and re-save
    metrics['final_test_accuracy'] = test_accuracy
    torch.save(metrics, 'linear_eval_metrics.pth') 
    print("Linear evaluation metrics updated with final test accuracy.")


if __name__ == "__main__":
    perform_linear_evaluation()

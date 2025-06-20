import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import os
import math # For cosine annealing scheduler
import time # For tracking training time

# Import the custom dataset and model
from simsiam_data import SatelliteDataset, simsiam_transform_v1, simsiam_transform_v2
from simsiam_model import SimSiam

# --- Configuration for Training ---
# Adjust these parameters as needed for your specific setup and desired training duration
BATCH_SIZE = 32 
NUM_EPOCHS = 200 # Total epochs for pre-training
LEARNING_RATE = 0.05 # Base learning rate for cosine schedule
WEIGHT_DECAY = 1e-4 # Standard for SimSiam
MOMENTUM = 0.9     # Standard for SimSiam
WARMUP_EPOCHS = 10 # Number of epochs for linear warmup
IMAGE_FOLDER_PATH = 'barlow_data/train' # Path to your generated PNG images
NUM_CHANNELS = 3 # Your PNGs are RGB (3 channels)
PROJECTION_DIM = 2048
PREDICTION_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# --- 1. Loss Function ---
def cosine_similarity_loss(p, z):
    """
    Calculates the negative cosine similarity loss for SimSiam.
    
    Args:
        p (torch.Tensor): Output of the predictor.
        z (torch.Tensor): Stop-gradient target representation (output of the projector).
        
    Returns:
        torch.Tensor: The mean negative cosine similarity.
    """
    # L2-normalize the vectors
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    
    # Return the negative cosine similarity.
    return -(p * z).sum(dim=1).mean()

# --- Cosine Annealing Learning Rate Scheduler with Warmup ---
def adjust_learning_rate(optimizer, base_lr, epoch, total_epochs, warmup_epochs, initial_warmup_lr_scale=0.1):
    """Decay the learning rate with half-cycle cosine annealing."""
    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (initial_warmup_lr_scale + (1 - initial_warmup_lr_scale) * (epoch / warmup_epochs))
    else:
        # Cosine annealing
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# --- 2. Training Setup ---
def train_simsiam():
    # Initialize metrics dictionary and lists *before* the loop
    metrics = {
        'epoch_losses': [],
        'epoch_learning_rates': [],
        'total_training_time_seconds': None, # Will be filled if run completes
    }

    print("Initializing dataset...")
    # Load your dataset with the two different transform pipelines
    dataset = SatelliteDataset(IMAGE_FOLDER_PATH, transform=simsiam_transform_v1, transform_v2=simsiam_transform_v2)
    # Using drop_last=True for BatchNorm stability in SimSiam
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, 
                            num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0,
                            pin_memory=True) # pin_memory can speed up data transfer to GPU
    
    print(f"Loaded {len(dataset)} images into dataset.")
    print(f"Number of batches per epoch: {len(dataloader)}")

    print("Initializing model...")
    # Initialize the base encoder (ResNet-50)
    base_encoder = models.resnet50(weights='IMAGENET1K_V1') # Using ResNet-50 now
    
    # Adapt the first convolutional layer if input channels don't match (should be 3 for RGB)
    if base_encoder.conv1.in_channels != NUM_CHANNELS:
        print(f"Adjusting base_encoder.conv1 from {base_encoder.conv1.in_channels} to {NUM_CHANNELS} channels.")
        original_conv1 = base_encoder.conv1
        base_encoder.conv1 = nn.Conv2d(
            NUM_CHANNELS,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

    # Initialize the SimSiam model
    model = SimSiam(base_encoder, proj_dim=PROJECTION_DIM, pred_dim=PREDICTION_DIM)
    model.to(DEVICE) # Move model to GPU if available

    # Optimizer (SGD with momentum is standard for SimSiam)
    optimizer = optim.SGD(model.parameters(), 
                          lr=LEARNING_RATE, # This will be the base_lr for the scheduler
                          momentum=MOMENTUM, 
                          weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")
    start_time = time.time() # Record start time

    for epoch in range(NUM_EPOCHS):
        # Adjust learning rate for the current epoch
        current_lr = adjust_learning_rate(optimizer, LEARNING_RATE, epoch, NUM_EPOCHS, WARMUP_EPOCHS)
        
        total_loss = 0
        
        # Use tqdm for a nice progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (LR: {current_lr:.6f})")
        for view1, view2 in pbar:
            # Move data to the appropriate device (GPU/CPU)
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            p1, z2, p2, z1 = model(view1, view2)
            
            # Calculate the symmetrized loss
            loss = (cosine_similarity_loss(p1, z2) + cosine_similarity_loss(p2, z1)) / 2
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Store metrics for plotting
        metrics['epoch_losses'].append(avg_loss)
        metrics['epoch_learning_rates'].append(current_lr)
        
        # Save metrics after *each* epoch
        torch.save(metrics, 'pretrain_metrics.pth')

        # Optional: Save model checkpoint periodically
        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_path = f"simsiam_resnet50_epoch_{epoch+1}.pth" # Changed checkpoint name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
    
    end_time = time.time() # Record end time
    total_training_time = end_time - start_time # Calculate total time
    print(f"\nTraining complete! Total training time: {total_training_time:.2f} seconds") 
    
    print(f"Final model saved to simsiam_resnet50_epoch_{NUM_EPOCHS}.pth")

    # Final save with total time (overwrites previous)
    metrics['total_training_time_seconds'] = total_training_time 
    torch.save(metrics, 'pretrain_metrics.pth')
    print("Pre-training metrics (final) saved to pretrain_metrics.pth")


if __name__ == "__main__":
    train_simsiam()
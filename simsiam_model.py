import torch
import torch.nn as nn
import torchvision.models as models

class SimSiam(nn.Module):
    def __init__(self, base_encoder, proj_dim=2048, pred_dim=512):
        """
        Initializes the SimSiam model.

        Args:
            base_encoder (nn.Module): The backbone model (e.g., ResNet).
                                      Its final fully connected layer will be removed.
            proj_dim (int): Dimension of the output of the projection head.
            pred_dim (int): Dimension of the intermediate layer of the prediction head.
        """
        super().__init__()
        
        # 1. Backbone (e.g., a ResNet-50)
        # We remove the final fully connected layer (classifier) of the pre-trained model.
        # This allows us to use the features extracted before classification.
        self.backbone = nn.Sequential(*list(base_encoder.children())[:-1])
        
        # Get the number of features from the backbone's last layer
        # This assumes the base_encoder has a .fc attribute (like ResNet).
        # If using a different backbone, this part might need adjustment.
        num_backbone_features = base_encoder.fc.in_features
        
        # 2. Projection Head (MLP - Multi-Layer Perceptron)
        # This projects the backbone's output into a higher-dimensional space.
        # It's a 3-layer MLP: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> Linear
        # The original SimSiam paper uses a 3-layer MLP with BatchNorm and ReLU.
        self.projector = nn.Sequential(
            nn.Linear(num_backbone_features, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True), # inplace=True saves memory by modifying the input directly
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True), # Second ReLU as per SimSiam paper (sometimes just 2 layers are used for simplicity, but 3 is typical)
            nn.Linear(proj_dim, proj_dim, bias=False), # Final linear layer
            nn.BatchNorm1d(proj_dim, affine=False) # No affine transformation here (no learnable scale/bias)
        )
        
        # 3. Prediction Head (MLP)
        # This head takes the output of the projector and predicts the target representation.
        # It's typically a 2-layer MLP (or 3 if you count the final linear layer as one)
        # The output dimension matches the projection head's output (proj_dim).
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim) # Output has same dim as projector's final output
        )

    def forward(self, x1, x2):
        """
        Forward pass for SimSiam.
        Takes two augmented views of an image and computes outputs for loss.
        """
        # Pass view1 and view2 through the encoder (backbone + projector)
        # .flatten(1) flattens all dimensions except the batch dimension (dim 0)
        # This is needed because the backbone's output before the FC layer (e.g., ResNet's avgpool output)
        # is typically (batch_size, num_features, 1, 1). Flattening makes it (batch_size, num_features).
        z1 = self.projector(self.backbone(x1).flatten(1))
        z2 = self.projector(self.backbone(x2).flatten(1))
        
        # Pass z1 and z2 through the predictor
        # p1 predicts z2, and p2 predicts z1
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Apply stop-gradient to the target representations (z2 and z1 for p1 and p2 respectively)
        # z.detach() means no gradients will flow back through z. This is the core
        # mechanism of SimSiam that prevents collapse without negative samples.
        return p1, z2.detach(), p2, z1.detach()

# --- How to create and test the model (for verification) ---
if __name__ == "__main__":
    # Define the number of input channels for your backbone.
    # Since your PNGs are RGB, this is 3.
    num_channels = 3 
    
    # 1. Load a base encoder (e.g., ResNet-18 or ResNet-50)
    # We use pre-trained weights for general image features, but it's not strictly necessary for SimSiam.
    # SimSiam works well even with randomly initialized backbones.
    # Using ResNet-18 for faster testing; you might switch to ResNet-50 later for better performance.
    #base_encoder = models.resnet18(weights='IMAGENET1K_V1') # Using pre-trained weights
    base_encoder = models.resnet50(weights='IMAGENET1K_V1') # Using pre-trained weights
    
    # Modify the first convolutional layer to accept 'num_channels' inputs
    # ResNet's default `conv1` expects 3 input channels. We need to adapt it
    # if your actual input has a different number of channels (which for your RGB PNGs, it's 3, so this is fine).
    # If your original Zarr data had more channels and you wanted to use them directly,
    # this part would become crucial. For now, it's a good practice to include.
    if base_encoder.conv1.in_channels != num_channels:
        print(f"Adjusting base_encoder.conv1 from {base_encoder.conv1.in_channels} to {num_channels} channels.")
        original_conv1 = base_encoder.conv1
        # Create a new conv1 layer with the correct number of input channels
        base_encoder.conv1 = nn.Conv2d(
            num_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # Optional: Initialize the new conv1 layer.
        # If the number of channels matches (3 in our case), we'd keep the original weights.
        # If it changes, you might want to average/sum original weights or just use new random ones.
        # For 3 channels, no specific action is needed here if base_encoder.conv1 is already 3.
        
    # Now, create the SimSiam model
    model = SimSiam(base_encoder, proj_dim=2048, pred_dim=512)
    print("\nSimSiam model created successfully!")
    print(model)

    # --- Test the model with dummy input ---
    print("\nTesting model with dummy input...")
    # Create dummy input data matching the expected shape: (batch_size, channels, height, width)
    dummy_input1 = torch.randn(2, num_channels, 224, 224) # Batch size 2, 3 channels, 224x224
    dummy_input2 = torch.randn(2, num_channels, 224, 224)

    p1, z2, p2, z1 = model(dummy_input1, dummy_input2)

    print(f"Shape of p1: {p1.shape}") # Should be [batch_size, proj_dim] e.g., [2, 2048]
    print(f"Shape of z2: {z2.shape}") # Should be [batch_size, proj_dim] e.g., [2, 2048]
    print(f"z2 requires grad: {z2.requires_grad}") # Should be False due to .detach()
    print(f"z1 requires grad: {z1.requires_grad}") # Should be False due to .detach()
    print("Model forward pass successful!")

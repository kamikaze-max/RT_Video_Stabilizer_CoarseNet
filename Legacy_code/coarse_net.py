# test_dummy_coarsenet.py

import torch
from models.coarse_net import CoarseNet

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create dummy optical flow input [batch_size, 2, height, width]
dummy_flow = torch.randn(1, 2, 256, 256).to(device)

# Initialize model
model = CoarseNet().to(device)
model.eval()

# Run forward pass
with torch.no_grad():
    output = model(dummy_flow)

print(f"Output transformation params: {output}")
print(f"Output shape: {output.shape}")  # Expect [1, 3]


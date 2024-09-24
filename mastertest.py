import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset 

class NiftiDataset(Dataset):
    def __init__(self, root_dir, limit=1):
        self.root_dir = root_dir
        self.case_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.case_folders = self.case_folders[:limit]

    def __len__(self):
        return len(self.case_folders)

    def __getitem__(self, idx):
        case_folder = self.case_folders[idx]
        image_file = os.path.join(self.root_dir, case_folder, 'imaging.nii.gz')
        mask_file = os.path.join(self.root_dir, case_folder, 'segmentation.nii.gz')

        # Load the image and mask
        image = nib.load(image_file).get_fdata()
        mask = nib.load(mask_file).get_fdata()

        # Convert image and mask to torch tensors, cast image to float32 and mask to long
        image = torch.from_numpy(image).float()  # Ensure float32 for the image
        mask = torch.from_numpy(mask).long()     # Mask should remain long for segmentation

        # Add a channel dimension to the image (1, depth, height, width)
        image = image.unsqueeze(0)

        return image, mask

        
# Set the root directory
root_dir = "/Users/daleelahmed/Desktop/kits19/data"

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim 

# Double convolution block using Conv3d
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),  # Use BatchNorm3d for 5D input
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),  # Use BatchNorm3d for 5D input
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # Use MaxPool3d for 3D data

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))

        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Define your NiftiDataset, UNet3D, and other components here

# Set hyperparameters
learning_rate = 1e-4
num_epochs = 1
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
model = UNet3D(in_channels=1, out_channels=1).to(device)  # 1 input channel (grayscale), 1 output channel (binary mask)
criterion = nn.BCEWithLogitsLoss()  # Assuming binary segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set the root directory
root_dir = "/Users/daleelahmed/Desktop/kits19/data"
nifti_dataset = NiftiDataset(root_dir=root_dir, limit=1)
train_loader = DataLoader(nifti_dataset, batch_size=batch_size, shuffle=True)

# Define the training loop function
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Make sure masks are in the right format (binary segmentation)
            masks = masks.float().unsqueeze(1)  # Add channel dimension to masks if needed
            
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("Training complete!")

# Start training the model
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

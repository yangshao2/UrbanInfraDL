#!/usr/bin/env python
"""
Script for semantic segmentation training using U-Net.

This script loads TIFF images and their corresponding label images, preprocesses them,
and trains a U-Net model for semantic segmentation. The dataset is split into
training and validation sets, and the best model checkpoint (based on validation loss)
is saved. After training, the script uses the best model to predict segmentation masks
for the input images and saves the results.

Requirements:
  - Python 3.x
  - PyTorch
  - rasterio
  - numpy
  - tqdm

Usage:
  Adjust the directory paths as needed, then run:
    python unet_trainer.py
"""

import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Dataset Class
class CustomTiffDataset(Dataset):
    """
    Custom dataset for reading TIFF images and corresponding label images.
    Assumes label filenames are derived by replacing 'image_patch_' with 'label_patch_'.
    """
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        if not self.images:
            raise ValueError(f"No .tif files found in the directory: {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.image_dir, img_name)
        with rasterio.open(image_path) as src_image:
            image = src_image.read().astype(np.float32)  # (C, H, W)
            # Normalize image to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = torch.tensor(image, dtype=torch.float32)
        
        # Derive corresponding label filename
        label_name = img_name.replace('image_patch_', 'label_patch_')
        label_path = os.path.join(self.label_dir, label_name)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} not found for image {img_name}")
        with rasterio.open(label_path) as src_label:
            label = src_label.read(1).astype(np.int64)  # (H, W)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# U-Net Model Definition
class UNet(nn.Module):
    """
    U-Net model for semantic segmentation.
    """
    def __init__(self, in_channels=3, num_classes=3):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # Resize outputs to match label dimensions if needed
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def predict_and_save(image_path, model, device, output_dir=None):
    """
    Given an image, predict the segmentation mask and save the result as a TIFF file.
    """
    model.eval()
    with torch.no_grad():
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            meta = src.meta.copy()
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0).to(device)
        output = model(image)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy().squeeze()
        meta.update({
            'count': 1,
            'dtype': 'uint8'
        })
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(image_path).replace('image_patch_', 'predicted_'))
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(pred_mask.astype(np.uint8), 1)
            print(f"Saved predicted mask to {output_path}")

def process_folder(input_dir, model, device, output_dir=None):
    """
    Process all TIFF images in the input directory for prediction.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            image_path = os.path.join(input_dir, filename)
            predict_and_save(image_path, model, device, output_dir)

def main():
    # Define paths (adjust these paths to your environment)
    image_dir = '/home/yshao/UrbanInfraDL/training/images'
    label_dir = '/home/yshao/UrbanInfraDL/training/labels'
    checkpoint_dir = '/home/yshao/UrbanInfraDL'
    output_dir = '/home/yshao/UrbanInfraDL/predictions'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create dataset and split into training and validation sets (80/20 split)
    full_dataset = CustomTiffDataset(image_dir, label_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training Loop
    num_epochs = 20
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_unet.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

    print("Training complete!")

    # Load the best model for predictions and process the folder
    model.load_state_dict(torch.load(best_model_path))
    process_folder(image_dir, model, device, output_dir=output_dir)

if __name__ == "__main__":
    main()

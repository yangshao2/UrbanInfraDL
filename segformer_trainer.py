#!/usr/bin/env python
"""
Script for semantic segmentation using a modified SegFormer model.

This script loads TIFF images and corresponding label images, extracts patches using a custom dataset,
and trains a semantic segmentation model based on a pre-trained SegFormer architecture. The script:
  - Ensures reproducibility by setting manual seeds.
  - Loads images and labels using a custom Dataset.
  - Modifies the SegFormer decoder to output a custom number of classes.
  - Splits the dataset into training and validation sets.
  - Trains and validates the model, saving the best checkpoint based on validation loss.

Requirements:
  - Python 3.x
  - PyTorch
  - Transformers (from Hugging Face)
  - rasterio
  - numpy
  - tqdm

Usage:
  Ensure that the image_dir and label_dir paths are correctly set, then run:
    python patch_extractor.py
"""

import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Load SegFormer Image Processor from pretrained checkpoint
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

# Dataset Class
class CustomTiffDataset(Dataset):
    """
    Custom dataset for reading TIFF images and corresponding label images.
    Assumes label filenames are derived by replacing 'image_patch_' with 'label_patch_' in the image filename.
    """
    def __init__(self, image_dir, label_dir, image_processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_processor = image_processor
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        if not self.images:
            raise ValueError(f"No .tif files found in the directory: {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.image_dir, img_name)
        with rasterio.open(image_path) as src_image:
            # Read image data and convert to (H, W, C)
            image = src_image.read().transpose(1, 2, 0).astype(np.float32)

        # Derive corresponding label filename
        label_name = img_name.replace('image_patch_', 'label_patch_')
        label_path = os.path.join(self.label_dir, label_name)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} not found for image {img_name}")
        with rasterio.open(label_path) as src_label:
            # Read the first channel for label (H, W)
            label = src_label.read(1).astype(np.int64)

        # Process image using the SegFormer image processor (includes normalization/rescaling)
        encoded_inputs = self.image_processor(image, return_tensors="pt", do_rescale=True)
        image_tensor = encoded_inputs["pixel_values"].squeeze(0)  # (C, H, W)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

# Model Class
class SegFormerModel(nn.Module):
    """
    A wrapper around the SegFormerForSemanticSegmentation model with a modified decoder head
    for a custom number of segmentation classes.
    """
    def __init__(self, num_classes):
        super(SegFormerModel, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        in_channels = self.model.decode_head.classifier.in_channels
        # Replace the classifier with a new convolutional layer to match the desired number of classes
        self.model.decode_head.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.model.decode_head.classifier.weight)
        nn.init.zeros_(self.model.decode_head.classifier.bias)
        self.model.config.num_labels = num_classes

    def forward(self, x):
        return self.model(x).logits

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        # Resize outputs to match label dimensions
        outputs = torch.nn.functional.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
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
            outputs = torch.nn.functional.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    # Define paths (adjust these to your system)
    image_dir = '/home/yshao/UrbanInfraDL/training/images'
    label_dir = '/home/yshao/UrbanInfraDL/training/labels'
    checkpoint_dir = "/home/yshao/UrbanInfraDL"
    #output_dir = '/media/newhd/yshao/bog/predictions'  # For future prediction outputs

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create dataset and split into training and validation sets
    full_dataset = CustomTiffDataset(image_dir, label_dir, image_processor)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegFormerModel(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10  # Adjust the number of epochs as needed
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_segformer.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()

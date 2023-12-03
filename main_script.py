# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:14:04 2023

@author: SABARI
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data_generator import ClassificationDataset  # Import your dataset module
from model import setup_model  # Import your model module
from train import train_batch,evaluate_batch
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 10

# Create instances of the dataset and dataloaders
train_dataset = ClassificationDataset(train=True, transform=transforms.ToTensor())  # Modify this line as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ClassificationDataset(train=False, transform=transforms.ToTensor())  # Modify this line as needed
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, criterion, and optimizer
model, criterion, optimizer = setup_model(num_classes=len(train_dataset.classes), device=device)

best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Training
    train_batch(train_loader, model, optimizer, device)
    
    # Evaluation
    outputs, targets, best_val_loss, val_loss = evaluate_batch(val_loader, model, best_val_loss, device)
    
    print(f"Validation Loss: {val_loss}")

# Save the final model
torch.save(model.state_dict(), "/content/drive/My Drive/WACV2023/final_model.pth")

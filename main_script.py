# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:14:04 2023

@author: SABARI
"""
import pandas as pd
import torch
from data_generator import ClassificationDataset  # Import your dataset module
from model import setup_model  # Import your model module
from train import train_batch, evaluate_batch  # Import your training functions
import numpy as np
from sklearn import metrics
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 100

# Load CSV file containing image names and labels
train_data_csv = pd.read_csv('/content/drive/My Drive/WACV2023/combined_data.csv')

# Split image names for training and validation
train_label_img_name = train_data_csv['image_name'][0:97669].values
valid_label_img_name = train_data_csv['image_name'][97669:].values

# Define the labels and their corresponding columns
label_col = ['Age-Young', 'Age-Adult', 'Age-Old', 'Gender-Female',
             'Hair-Length-Short', 'Hair-Length-Long', 'Hair-Length-Bald',
             'UpperBody-Length-Short', 'UpperBody-Color-Black',
             'UpperBody-Color-Blue', 'UpperBody-Color-Brown',
             'UpperBody-Color-Green', 'UpperBody-Color-Grey',
             'UpperBody-Color-Orange', 'UpperBody-Color-Pink',
             'UpperBody-Color-Purple', 'UpperBody-Color-Red',
             'UpperBody-Color-White', 'UpperBody-Color-Yellow',
             'UpperBody-Color-Other', 'LowerBody-Length-Short',
             'LowerBody-Color-Black', 'LowerBody-Color-Blue',
             'LowerBody-Color-Brown', 'LowerBody-Color-Green',
             'LowerBody-Color-Grey', 'LowerBody-Color-Orange',
             'LowerBody-Color-Pink', 'LowerBody-Color-Purple', 'LowerBody-Color-Red',
             'LowerBody-Color-White', 'LowerBody-Color-Yellow',
             'LowerBody-Color-Other', 'LowerBody-Type-Trousers&Shorts',
             'LowerBody-Type-Skirt&Dress', 'Accessory-Backpack', 'Accessory-Bag',
             'Accessory-Glasses-Normal', 'Accessory-Glasses-Sun', 'Accessory-Hat']

# Extract labels for training and validation sets
train_label = train_data_csv[label_col][0:97669].values
valid_label = train_data_csv[label_col][97669:].values

# Construct full paths for images in training and validation sets
train_label_names = ["/content/new_dataset/copied_files/" + name for name in train_label_img_name]
valid_label_names = ["/content/new_dataset/copied_files/" + name for name in valid_label_img_name]

# Create instances of the dataset and dataloaders
train_dataset = ClassificationDataset(image_paths=train_label_names,
                                      targets=train_label,
                                      resize=(128, 64),
                                      augmentations=None,
                                      )

valid_dataset = ClassificationDataset(image_paths=valid_label_names,
                                      targets=valid_label,
                                      resize=(128, 64),
                                      augmentations=None,
                                      )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

# Initialize the model, criterion, and optimizer
model, criterion, optimizer = setup_model(num_classes=len(train_dataset.classes), device=device)

best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Training
    train_batch(train_loader, model, optimizer, device)
    
    # Evaluation   
    predictions, valid_targets,best_val_loss,val_loss = evaluate_batch(
        valid_loader, model,best_val_loss,device=device
        )
    #roc_auc = metrics.roc_auc_score(valid_targets, predictions)
    # Calculate accuracy
    valid_targets = np.array(valid_targets).flatten()
    predictions = np.array(predictions).flatten()
    predictions = np.uint8(predictions>0.5)
    accuracy = metrics.accuracy_score(valid_targets, predictions)

    # Calculate F1 score
    #f1 = metrics.f1_score(valid_targets, predictions, average='macro')
    f1 = metrics.f1_score(valid_targets, predictions,average='weighted')
    print(f"Validation Loss: {val_loss}")

# Save the final model
torch.save(model.state_dict(), "/content/drive/My Drive/WACV2023/final_model.pth")

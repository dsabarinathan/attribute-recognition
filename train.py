# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:12:11 2023

@author: SABARI
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn import metrics
import albumentations
from torch.utils.data import Dataset
# Define your ClassificationDataset class and other necessary classes/functions here

def train_batch(data_loader, model, optimizer, device):
    model.train()
    for data in data_loader:
        inputs = data["image"]
        targets = data["targets"]
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate_batch(data_loader, model, best_val_loss, device):
    checkpoint_filepath = "/content/drive/My Drive/WACV2023/"
    model.eval()
    final_targets = []
    final_outputs = []
    val_loss = 0

    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            output = model(inputs)
            cur_valid_loss = nn.BCEWithLogitsLoss()(output, targets)
            val_loss += cur_valid_loss.item()

            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            final_targets.extend(targets)
            final_outputs.extend(output)

    val_loss = val_loss / len(data_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_filepath + "DeepMAR_ResNet18_best_model.pth")
        print("Model saved!: "+checkpoint_filepath + "DeepMAR_ResNet18_best_model.pth")

    return final_outputs, final_targets, best_val_loss,val_loss



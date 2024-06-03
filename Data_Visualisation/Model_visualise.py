import torch
import torch.nn as nn
from torchviz import make_dot

import numpy as np

import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import datetime
import cv2
import seaborn as sns
import traceback
import random

class Raw_stream(nn.Module):
    def __init__(self):
        super(Raw_stream, self).__init__()
        # Commented for Say Version
        # self.conv1 = nn.Conv3d(1, 128, (3, 3, 3), padding = 1)
        # self.conv2 = nn.Conv3d(128, 128, (3, 3, 3), padding = 1)
        # self.conv3 = nn.Conv3d(128, 64, (3, 3, 3), padding = 1)
        
        # self.pool = nn.MaxPool3d(2)

        # # self.fc1 = nn.Linear(3072, 64)
        # self.fc1 = nn.Linear(64 * 2 * 4 * 6, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 254)
        # self.fc4 = nn.Linear(254, 2) 

        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 128, (3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 128, (3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2) 


    def forward(self, x):
        # print(f"shape: {x.shape}")
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
    
        return x
    
class OF_Stream(nn.Module):
    def __init__(self):
        super(OF_Stream, self).__init__()
        # self.conv1 = nn.Conv3d(2, 128, (3, 3, 3), padding = 1)
        # self.conv2 = nn.Conv3d(128, 128, (3, 3, 3), padding = 1)
        # self.conv3 = nn.Conv3d(128, 64, (3, 3, 3), padding = 1)
        
        # self.pool = nn.MaxPool3d(2)

        # self.fc1 = nn.Linear(64 * 2 * 4 * 6, 64)
        # # self.fc1 = nn.Linear(64 * 48, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 254)
        # self.fc4 = nn.Linear(254, 2) 

        
        # Convolutional layers
        self.conv1 = nn.Conv3d(2, 64, (3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2) 

    def forward(self, x):
        # print(f"shape: {x.shape}")
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)   
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class TwoStreamConv(nn.Module):
    def __init__(self):
        super(TwoStreamConv, self).__init__()
        # Define the two streams
        self.stream1 = Raw_stream()
        self.stream2 = OF_Stream()
        # self.stream1 = FallDetectionCNN_Say()
        # self.stream2 = FallDetectionCNN_Say()
        # Define the softmax layer
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x1, x2):
        # Forward pass through each stream
        x1 = self.stream1(x1)
        x2 = self.stream2(x2)

        # print(f"Size of x1 along dimension 0: {x1.size(0)}")
        # print(f"Size of x2 along dimension 0: {x2.size(0)}")    
        # Concatenate the outputs of two streams
        x = torch.cat((x1, x2), dim=1)
        # Final softmax layer
        x = self.fc2(x)
        return F.softmax(x, dim=1)  
        # return x  
    
# Example usage:
num_classes = 10  # Change this according to your dataset
model = TwoStreamConv()

# Create dummy inputs
raw_input = torch.randn(1, 3, 32, 32, 32)  # Assuming input size of 32x32x32 for raw stream
of_input = torch.randn(1, 2, 32, 32, 32)   # Assuming input size of 32x32x32 for optical flow stream

# Generate the visualization
output = model(raw_input, of_input)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("two_stream_cnn_architecture")

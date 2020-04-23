# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:10:11 2020

@author: Deeps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch.utils import data
from torchsummary import summary
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split as tts
import torch.nn.functional as F
from torchvision import datasets,models, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from data_gen import Dataset

# Load the dataset and train, val, test splits

num_epochs = 200
num_classes = 2
batch_size = 3
learning_rate = 0.0005
datacsv = pd.read_csv("data.csv")
cv_results = []
##################Create dataset generator##################################

for cv in range(6):
    x_train,x_test =tts(datacsv["image"],test_size=1/6, shuffle=True)
    partition = {'train':list(x_train),'validation':list(x_test)}
    labels = {}
    
    for k in range(len(datacsv['image'])):
        labels['%s'%datacsv['image'][k]] = datacsv['label'][k]
        
    #####################################################################
    
    training_set = Dataset(partition["train"],labels)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    
    validation_set = Dataset(partition['validation'], labels)
    validation_generator = data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    
    
    
    #######################################################
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: Design your own network, define layers here.
            # Here We provide a sample of two-layer fully-connected network.
            # Your solution, however, should contain convolutional layers.
            # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
            # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout
            # If you have many layers, consider using nn.Sequential() to simplify your code
            # self.fc1 = nn.Linear(28*28, 8) # from 28x28 input image to hidden layer of size 256
            # self.fc2 = nn.Linear(8,10) # from hidden layer to 10 class scores
            self.conv1 = nn.Conv2d(3, 16, 5) #18*128*128
            self.pool = nn.MaxPool2d(2, 2) #18*64*64
            self.conv2 = nn.Conv2d(16, 32, 5) #36*64*64
            self.fc1 = nn.Linear(32 * 29 * 29, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
            
    
        def forward(self,x):
            # TODO: Design your own network, implement forward pass here
            x = self.pool(F.relu(self.conv1(x))) #16*124*124 -> 16*62*62
            x = self.pool(F.relu(self.conv2(x))) #32*58*58 -> 32*29*29
            x = x.view(-1, 32 * 29 * 29)
            x = F.dropout(F.relu(self.fc1(x)))
            x = F.relu(self.fc2(x))
            out = F.sigmoid(self.fc3(x))
            return out
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
    model = Network().to(device)
    summary(model,(3,128,128))
    criterion = nn.CrossEntropyLoss() # Specify the loss layer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
    
    
    
    def train(model, training_generator, num_epoch = num_epochs): # Train the model
        print("Start training...")
        model.train() # Set the model to training mode
        for i in range(num_epoch):
            running_loss = []
            for batch, label in tqdm(training_generator):
                batch = batch.to(device)
                label = label.to(device)
                optimizer.zero_grad() # Clear gradients from the previous iteration
                pred = model(batch) # This will call Network.forward() that you implement
                loss = criterion(pred, label) # Calculate the loss
                running_loss.append(loss.item())
                loss.backward() # Backprop gradients to all tensors in the network
                optimizer.step() # Update trainable weights
            print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
        print("Done!")
    
    def evaluate(model, validation_generator): # Evaluate accuracy on validation / test set
        model.eval() # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.set_grad_enabled(True): # Do not calculate grident to speed up computation
            for batch, label in tqdm(validation_generator):
                batch = batch.to(device)
                label = label.to(device)
                pred = model(batch)
                correct += (torch.argmax(pred,dim=1)==label).sum().item()
                total+=batch_size
        acc = correct/total
        print("Evaluation accuracy: {}".format(acc))
        return acc
        
    train(model, training_generator, num_epochs)
    print("Evaluate on validation set...")
    result = evaluate(model, validation_generator)
    cv_results.append(result)

np.mean(cv_results)
result = evaluate(model, training_generator)

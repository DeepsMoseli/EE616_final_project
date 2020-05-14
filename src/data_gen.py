# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:42:18 2020

@Moseli
"""

########################dataset creation##################

import torch
from torch.utils import data
from functools import partial
from PIL import Image
from torchvision import transforms, datasets

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, size = (748,500)):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.size = size
        self.resize = (374,250)
        self.teeth_transform = transforms.Compose([
            transforms.Resize ( self.size , interpolation=2 ),
            transforms.CenterCrop(self.resize[0]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
            transforms.Normalize((0.5, 0.5, 0.5), (0.229, 0.224, 0.225)) # Normalize to zero mean and unit variance
            ])


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        #X = torch.load('Dataset/' + str(ID) + '.jpg', pickle_module=pickle)
        X = Image.open('Dataset/' + str(ID) + '.jpg')
        X = X.crop((0, 0, self.size[0], self.size[1])).convert('LA')
        X = self.teeth_transform(X)
        y = self.labels[str(ID)]

        return X, y
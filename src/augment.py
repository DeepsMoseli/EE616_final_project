# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:25:28 2020

@author: Deeps
"""

from PIL import Image
from torchvision import transforms
import pandas as pd

class Augmentations:
    def __init__(self,angles =[90,180, 270]):
        self.angles = angles
        
    def rotate_append(self,number,label):
        hold = []
        labels = [label for k in self.angles]
        img = Image.open('Dataset/' + str(number) + '.jpg')
        
        for k in range(len(self.angles)):
            imageName = 120+(len(self.angles)*(number-1))+(k+1)
            hold.append(imageName)
            img_rot = img.rotate(self.angles[k])
            img_rot.save('Dataset/' + str(imageName) + '.jpg')
        return pd.DataFrame({"image":hold,"label":labels})



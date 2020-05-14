# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:25:28 2020

@author: Deeps
"""

from PIL import Image,ImageFilter
from torchvision import transforms
import pandas as pd
import numpy as np

class Augmentations:
    def __init__(self,angles =[]):
        self.angles = angles
        
    def rotate_append(self,number,label):
        hold = []
        if int(label)==0:
            labels = [label for k in self.angles]
            img = Image.open('Dataset/' + str(number) + '.jpg')
            for k in range(len(self.angles)):
                imageName = 120+(len(self.angles)*(number-1))+(k+1)
                hold.append(imageName)
                img_rot = img.rotate(self.angles[k])
                image_rot_blurred = img_rot.filter(ImageFilter.GaussianBlur(radius = np.random.randint(4))) 
                image_rot_blurred.save('Dataset/' + str(imageName) + '.jpg')
            return pd.DataFrame({"image":hold,"label":labels})
        else:
            if np.random.random()>0.6:
                labels = [label for k in self.angles]
                img = Image.open('Dataset/' + str(number) + '.jpg')
                for k in range(len(self.angles)):
                    imageName = 120+(len(self.angles)*(number-1))+(k+1)
                    hold.append(imageName)
                    img_rot = img.rotate(self.angles[k])
                    image_rot_blurred = img_rot.filter(ImageFilter.GaussianBlur(radius = np.random.randint(4))) 
                    image_rot_blurred.save('Dataset/' + str(imageName) + '.jpg')
                return pd.DataFrame({"image":hold,"label":labels})



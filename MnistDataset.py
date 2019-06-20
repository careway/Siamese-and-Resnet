# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:19:15 2019

@author: s182119
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
from os.path import join
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image





class MNIST(dset.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,valid=False):
        super(MNIST,self).__init__(root, train, transform, target_transform, download)
        self.valid = valid
        self.equal = False
        

    def __getitem__(self, index):
        label = None
        img1 = None
        img2 = None
        img3 = None
        
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target2 = -1
        target3 = -1
        img1, target1 = super(MNIST,self).__getitem__(index)
        lend = len(self)
        idx = random.randint(0,lend-1)
        img2, target2 = super(MNIST,self).__getitem__(idx) 
        while not self.valid and target2 == 9 and self.equal!=(target1==target2):
            idx = random.randint(0,lend-1)
            img2, target2 = super(MNIST,self).__getitem__(idx) 
        while self.valid and self.equal!=(target2 ==9):
            idx = random.randint(0,lend-1)
            img2, target2 = super(MNIST,self).__getitem__(idx) 
            
        self.equal = False if self.equal else True
        """
        while (tarjet3 != None or tarjet3 == tarjet):
            idx = random.randint(0,lend)
            img3, target3 = self.data[idx], int(self.targets[idx])
        """
        label = 1. if target1 == target2 else 0.
        
        
        return img1,img2,label,target1.type(torch.LongTensor)
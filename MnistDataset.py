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
        tarjet2 = None
        tarjet3 = None
        img1, target = self.data[index], int(self.targets[index])
        lend = len(self.data)
        while (tarjet != tarjet2 or tarjet2 != None):
            idx = random.randint(0,lend)
            img2, target2 = self.data[idx], int(self.targets[idx])
        """
        while (tarjet3 != None or tarjet3 == tarjet):
            idx = random.randint(0,lend)
            img3, target3 = self.data[idx], int(self.targets[idx])
        """
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L') 
        #img3 = Image.fromarray(img3.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return img1,img2,tarjet
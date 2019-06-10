# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:52:32 2019

@author: s182119
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models




class SiameseNetwork(nn.Module):

    def __init__(self):
        self.conv1 = nn.Conv2d( 1, 64, 10)
        self.pool1  = nn.MaxPool2d (2)       

        self.conv1 = nn.Conv2d( 64, 128, 7)
        self.pool  = nn.MaxPool2d (2)       

        self.conv1 = nn.Conv2d( 128, 128, 4)
        self.pool  = nn.MaxPool2d (2)       

        self.conv1 = nn.Conv2d( 128, 256, 4)
        self.fc = nn.Linear(256*6*6,4096)
        
        self.fco = nn.Linear(96,1)

        
        
    def forward_one(self):
        o = F.relu(self.conv1(x))
        o = self.pool1(o)
        
        o = F.relu(self.conv2(o))
        o = self.pool2(o)
        
        o = F.relu(self.conv3(o))
        o = self.pool3(o)
        
        o = F.relu(self.conv4(o))
        o = self.pool4(o)
        
        o = F.relu(self.conv5(o))
        o = F.sigmoid(self.fc(o))
        
        
        return o
    
    def forward(self,x1,x2):
        f1 = forward_one(x1)
        f2 = forward_one(x2)
        dis = torch.abs(f1-f2)
        out = self.fco(dis)
        return f1,out
        
"""
class ResNet(nn.Module):
    def __init__(self,):
        super(SiameseNetowrk,self).__init__()
        
        
        
        
    def forward(self,):
        
        
        
        
class SiaResNetwork(nn.Module):
    def __init__(self,):
        super(SiameseNetwork,self).__init__()
        
        self.resnet18 = models.resnet180(pretrained=False)
        
        self.siamese = SiameseNetwork()

        
    def forward(self,x1,x2):
        out, features = self.siamese(x1,x2)
        
"""    
    
    

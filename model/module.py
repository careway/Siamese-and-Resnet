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
from math import log



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.pd = nn.PairwiseDistance()

    def forward(self, output1, output2, target, size_average=True):
        distances = self.pd(output1,output2)  # squared distances
        losses = (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class ContrastiveLossq(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLossq, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.conv1 = nn.Conv2d( 1, 64, 10)
        self.pool1  = nn.MaxPool2d (2)       

        self.conv2 = nn.Conv2d( 64, 128, 7)
        self.pool2  = nn.MaxPool2d (2)       

        self.conv3 = nn.Conv2d( 128, 128, 4)
        self.pool3  = nn.MaxPool2d (2)       

        self.conv4 = nn.Conv2d( 128, 256, 4)
        self.fc = nn.Linear(256*6*6,4096)
        
        self.fco = nn.Linear(4096,1)

        
        
    def forward_one(self,x):
        o = F.relu(self.conv1(x),2)
        o = self.pool1(o)
        
        o = F.relu(self.conv2(o),2)
        o = self.pool2(o)
        
        o = F.relu(self.conv3(o),2)
        o = self.pool3(o)
        
        o = F.relu(self.conv4(o),2)
        f = o.view(o.shape[0], -1)
        o = torch.sigmoid(self.fc(f))
        
        
        return o,f
    
    def forward(self,x1,x2):
        o1,f1 = self.forward_one(x1)
        o2,f2 = self.forward_one(x2)
        dis = torch.abs(o1-o2)
        out = self.fco(dis)
        return f1,f2,out
        
            
class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    """
    def __init__(self,n_input,n_output):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(n_input, n_output)  # Implement the Linear function
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd        
        
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class MNISTResNet(ResNet):
    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2]) # Based on ResNet1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)
        
class SiaResNetwork(nn.Module):

    def __init__(self,featSM=4096,featRN=512,n_classes=9216):
        super(SiaResNetwork,self).__init__()
        
        self.resnet18 = MNISTResNet()
        self.layer = self.resnet18._modules.get('avgpool')
        self.siamese = SiameseNetwork()
        self.my_embedding = torch.zeros(1,9216)
        self.svm = SVM(featSM+featRN,n_classes)
                
    def forward(self,x1,x2):
        features1,_,out1 = self.siamese(x1,x2)
        def fun(m, i, o): self.my_embedding.copy_(o.data)
        h = self.resnet18.register_forward_hook(fun)
        out2 = self.resnet18(x1)
        h.remove()
        
        out3 = self.svm(features1+self.my_embedding)
        return out1,out2,out3
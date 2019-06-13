# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:06:53 2019

@author: s182119
"""

import time
import random
import numpy as np

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import AverageMeter
import MnistDataset
import model.module as md
from matplotlib import pyplot as plt



#### Data creation
use_gpu = True


## Dataset
valid_size = 0.2
batch_size = 1
shuffle = True
num_workers = 2
pin_memory = True


routedt= "./dataset/"

transforms = transforms.Compose([transforms.Resize((105,105)),
                                 transforms.ToTensor()])



data = MnistDataset.MNIST(root=routedt,transform=transforms,download=True)

datatest = MnistDataset.MNIST(root=routedt,transform=transforms,train=False)

indices = list(range(len(data)))
split = int(np.floor(valid_size * len(data)))
train_idx, valid_idx = indices[split:], indices[:split]

train_loader = DataLoader(data,batch_size,shuffle,num_workers=num_workers,
                          pin_memory=pin_memory)
valid_loader = DataLoader(data,1,shuffle,num_workers=num_workers,
                          pin_memory=pin_memory)
                          
test_loader = DataLoader(datatest,1,shuffle,num_workers=num_workers,
                          pin_memory=pin_memory)

## Model, loss and optimicer
model = md.SiaResNetwork()

criterion = F.binary_cross_entropy_with_logits
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=6e-5)
if use_gpu:
    model.cuda()


def create_trainer_dset():
    xs,x1s = [],[]
    lbls,ys = [],[]

    for i,(x,x1,y,label) in enumerate(train_loader):
        if y not in ys:
            xs.append(x)
            x1s.append(x1)
            ys.append(y)
            lbls.append(label)
            
        if len(ys) == 10:
            break
    return zip(xs,x1s,lbls,ys)

def train(epoch):
    plot = False
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    tic = time.time()
    with tqdm(total=10) as pbar:
        for i, (x,x1,y,label) in enumerate(create_trainer_dset()):
            if use_gpu:
                x, x1, y = x.cuda(), x1.cuda(), y.cuda()
            x, x1, y = Variable(x),Variable(x1), Variable(y)
            
            out1,out2,out3 = model(x,x1)
            loss_siamese = criterion(out1[0].cpu(),y.type(torch.FloatTensor))
            _,loss_resnet = torch.max(out2)==label
            _,loss_siares = torch.max(out3)==label
            loss = loss_siamese + loss_resnet + loss_siares
            losses.update(loss)
            #accs.update(sum(y==y_pred))
            
            #print
            if plot:
                ax1 = plt.subplot(10,2,i*2+1)
                ax2 = plt.subplot(10,2,i*2+2)
                ax1.imshow(x.cpu()[0,0])
                ax2.imshow(x1.cpu()[0,0])
                ax1.axis('off')
                ax2.axis('off')
                plt.title("diff:{:.3f}".format(y_pred.item()),loc='right')

            # compute gradients and update ADAM
            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)
            
            #pbar.set_description("   Train: {:.1f}s - loss: {:.3f}".format((toc-tic),
             #   loss.item())
            #)
            pbar.update(batch_size)
    if plot:
        plt.subplots_adjust(hspace=0)
        plt.show()
    return losses.avg,accs.avg
            
def evaluate (epoch):
    plot = False
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    tic = time.time()
    
    with tqdm(total=len(datatest)) as pbar:
        for i,(x,x1,label,y) in enumerate(test_loader):
            if use_gpu:
                x, x1, y = x.cuda(), x1.cuda(), y.cuda()
            x, x1, y = Variable(x),Variable(x1), Variable(y)

            f1,f2,y_pred = model(x,x1)
            loss = criterion(y_pred[0].cpu(),y.type(torch.FloatTensor))
            losses.update(loss.item())
            #accs.update(sum(y==y_pred))

                        
            
            
            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)
            
           # pbar.set_description("   Train: {:.1f}s - loss: {:.3f}".format((toc-tic),
            #    loss.item())
            #)
            pbar.update(batch_size)
            torch.cuda.empty_cache()
    return losses.avg,accs.avg

epochs = 60
for i in range(epochs):
    
    model.train()
    train_loss,train_acc = train(i)
    print("Train epoch {}: loss = {:.3f} - acc = {:.3f}".format(i,train_loss,train_acc))
    model.eval()
    valid_loss,valid_acc = evaluate(i)
    print("Valid epoch {}: loss = {:.3f} - acc = {:.3f}".format(i,valid_loss,valid_acc))
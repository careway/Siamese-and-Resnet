# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:06:53 2019

@author: s182119
"""

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
import torch.nn as nn

import numpy as np

import random



#### Data creation


## Dataset
valid_size = 0.2
batch_size = 60
shuffle = True
num_workers = 2
pin_memory = True


routedt= "./dataset/"

transforms = transforms.Compose([transforms.Resize(105,105),
                                 transforms.ToTensor()])



data = datasets.Omniglot(root=routedt,transform=transforms,download=True)


indices = list(range(len(data)))
split = int(np.floor(valid_size * len(data)))
train_idx, valid_idx = indices[split:], indices[:split]
        
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(data,batch_size,shuffle,train_sampler,num_workers,
                          pin_memory)
valid_loader = DataLoader(data,batch_size,shuffle,valid_sampler,num_workers,
                          pin_memory)


def create_trainer_dset():
    xs,x1s = [],[]
    ys = []
    for i,(x,x1,y) in enumerate(train_loader):
        if y is not in ys:
            xs.append(x)
            x1s.append(x1)
            ys.append(y)
        if len(ys) == 10:
            break
    return xs,x1s,ys

def train(epoch):
    plot = False

    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    tic = time.time()
    
    with tqdm(total=num_train*(1-valid_size)) as pbar:
        for i, (x,x1,y) in enumerate(create_trainer_dset()):
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            
            y = model(x,x1,y)
            
            
            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)
            
            pbar.set_description("   Train: {:.1f}s - loss: {:.3f} - acc: {:.3f}".format((toc-tic),
                loss.item(), acc.item())
            )
            pbar.update(batch_size)
            
        

for i in range(epochs):
    
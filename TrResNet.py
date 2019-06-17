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
import torchvision.models as models



#### Variables set
use_gpu = True
valid_size = 0.2
batch_size = 60
shuffle = True
num_workers = 2
pin_memory = True
test = False
# Train loop
plot = False
epochs = 60
train_patience=20
best_valid_loss=1
counter=0
### Data Creation
#Route
routedt= "./dataset/"
#Transformations
transforms = transforms.Compose([transforms.Resize((105,105)),
                                 transforms.ToTensor()])


#Loading Data
if not test:
    data = MnistDataset.MNIST(root=routedt,transform=transforms,download=True)
else:
    data = MnistDataset.MNIST(root=routedt,transform=transforms,train=False)

#Parsing data for one shot training
#Training dataloader
idx = data.train_labels!=9
idx9 = data.train_labels==9
train_sampler = SubsetRandomSampler(torch.cat((idx.nonzero().flatten(),idx9.nonzero()[0])))

train_loader = DataLoader(data,batch_size=batch_size,sampler=train_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)
#Valid dataloader
valid_sampler = SubsetRandomSampler(idx9.nonzero().flatten())
valid_loader = DataLoader(data,batch_size=batch_size,sampler=valid_sampler,num_workers=num_workers,
                          pin_memory=pin_memory)
                          

#### Model, loss and optimicer
model = mod()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-5)

if use_gpu:
    model.cuda()

def train(epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    tic = time.time()
    #fig, big_axes = plt.subplots( figsize=(15.0, 15.0) , nrows=10, ncols=1) 
    with tqdm(total=len(train_loader)*batch_size) as pbar:
        for i, (x,_,_,y) in enumerate(train_loader):
            if use_gpu:
                x,y = x.cuda(), y.cuda()
            x, y = Variable(x),Variable(y)
            
            y_pred = model(x)

            loss = criterion(y_pred,  y)
            losses.update(loss.item())
            #accs.update(sum(y==y_pred))
            predicted = torch.max(y_pred, 1)[1]
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))
            accs.update(acc)
            # compute gradients and update ADAM
            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)
            
            pbar.set_description("   Train: {:.1f}s".format(toc-tic))
            pbar.update(batch_size)
    return losses.avg, accs.avg
            
def evaluate (epoch):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    tic = time.time()
    corr = 0
    with tqdm(total=len(valid_loader)*batch_size) as pbar:
        for i,(x,_,_,y) in enumerate(valid_loader):
            if use_gpu:
                x,  y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            
            y_pred = model(x)
            
            loss = criterion(y_pred,y)
            losses.update(loss.item())
            #accs.update(sum(y==y_pred))
            predicted = torch.max(y_pred, 1)[1]
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))
            accs.update(acc)
            if plot:
                plt.figure()
                ax1 = plt.subplot(1,2,1)
                ax2 = plt.subplot(1,2,2)
                ax1.imshow(x[0][0].cpu())
                ax2.imshow(x1[0][0].cpu())
                ax1.axis('off')
                ax2.axis('off')
                plt.title("diff:{:.3f} y:{}".format(y_pred[0].item(),y[0]),loc='right')
                plt.show()
            
            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)
            corr += correct.sum()
            pbar.set_description("   Valid: {:.1f}s corr:{:.1f}".format(toc-tic,corr))
            pbar.update(batch_size)
    return losses.avg,accs.avg


for i in range(epochs):
    
    model.train()
    train_loss,train_acc = train(i)
    model.eval()
    valid_loss,valid_acc = evaluate(i)
    # # reduce lr if validation loss plateaus
    # self.scheduler.step(valid_loss)

    is_best = valid_loss < best_valid_loss
    msg = "Epoch {}: Train loss = {:.3f} - acc = {:.3f} - Valid loss = {:.3f} - acc = {:.3f}"
    if is_best:
        model_save = model.state_dict()
        counter = 0
        msg += " [*]"
    print(msg.format(i, train_loss, train_acc,valid_loss,valid_acc))

    # check for improvement
    if not is_best:
        counter += 1
    if counter > train_patience:
        print("[!] No improvement in a while, stopping training.")
        break
    best_valid_loss = min(valid_loss, best_valid_loss)    
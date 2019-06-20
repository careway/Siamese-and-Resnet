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
import itertools
from matplotlib import pyplot as plt



#### Variables set
use_gpu = True
valid_size = 0.2
batch_size = 60
shuffle = True
num_workers = 1
pin_memory = True
test = False
counter = 0
# Train loop
plot = False
epochs = 200
train_patience=60
best_valid_loss=20
#optim
lr=3e-4
wd=6e-5


### Data Creation
#Route
routedt= "./dataset/"
#Transformations
transforms = transforms.Compose([transforms.Resize((105,105)),
                                 transforms.ToTensor()])


#Loading Data
if not test:
    data = MnistDataset.MNIST(root=routedt,transform=transforms,download=True)
    dataval= MnistDataset.MNIST(root=routedt,transform=transforms,download=True,valid=True)
else:
    data = MnistDataset.MNIST(root=routedt,transform=transforms,train=False)

#Parsing data for one shot training
#Training dataloader
idx = data.train_labels!=9
train_sampler = SubsetRandomSampler(idx.nonzero().flatten())

train_loader = DataLoader(data,batch_size=batch_size,sampler=train_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)
#Valid dataloader
idx = dataval.train_labels==9
valid_sampler = SubsetRandomSampler(idx.nonzero().flatten())
valid_loader = DataLoader(dataval,batch_size=batch_size,sampler=valid_sampler,num_workers=num_workers,
                          pin_memory=pin_memory)
                          

## Model, loss and optimicer
model = md.SiaResNetwork()

criterion = nn.BCEWithLogitsLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
if use_gpu:
    model.cuda()
    

def train(epoch):
    plot = False
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    tic = time.time()
    with tqdm(total=len(train_loader)*batch_size) as pbar:
        for i, (x,x1,label,y) in enumerate(train_loader):
            if use_gpu:
                x, x1, y = x.cuda(), x1.cuda(), y.cuda()
            x, x1, y = Variable(x),Variable(x1), Variable(y)
            
          
            out1,fco = model(x,x1)
            
            loss_siamese = criterion(out1.flatten(),label.type(torch.FloatTensor).flatten().cuda())
            loss_final_class = criterion2(fco,y)
            
            loss = loss_siamese + loss_final_class
            losses.update(loss.item())
            #accs.update(sum(y==y_pred))
            predicted = torch.max(fco, 1)[1]
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))
            accs.update(acc)
            
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
            loss_siamese.backward(retain_graph=True)
            loss_final_class.backward()
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
    classification=[]
    with tqdm(total=len(valid_loader)*batch_size) as pbar:
        for i,(x,x1,label,y) in enumerate(valid_loader):
            if use_gpu:
                x, x1, y = x.cuda(), x1.cuda(), y.cuda()
            x, x1, y = Variable(x),Variable(x1), Variable(y)

            out1,fco = model(x,x1)
            
            loss_siamese = criterion(out1.flatten().cpu(),label.type(torch.FloatTensor).flatten())
            loss_final_class = criterion2(fco,y)
            
            loss = loss_siamese + loss_final_class
            losses.update(loss.item())
            #accs.update(sum(y==y_pred))
            predicted = torch.max(fco, 1)[1]
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))
            accs.update(acc.item())
            
            #loss graphic from siamese
            classification.append(list(zip(out1.flatten().cpu().tolist(),label.type(torch.FloatTensor).flatten().tolist())))
            
            
            
            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)
            
           # pbar.set_description("   Train: {:.1f}s - loss: {:.3f}".format((toc-tic),
            #    loss.item())
            #)
            pbar.update(batch_size)
    return losses.avg,accs.avg,classification

epochs = 60
for i in range(epochs):
    
    model.train()
    train_loss,train_acc = train(i)
    model.eval()
    valid_loss,valid_acc,clss = evaluate(i)
    
    rt = list(itertools.chain(*clss))
    x = [x[0] for x in rt]
    y = [x[1] for x in rt]
    x = np.array(x)
    idxb = [ys == 1. for ys in y]
    idxr = [ys == 0. for ys in y]    
    
    
    
    is_best = valid_loss < best_valid_loss
    msg = "Epoch {}: Train: loss = {:.3f} acc = {:.3f} - Valid: loss = {:.3f} acc={:.3f}"
    if is_best:
        model_save = model.state_dict()
        counter = 0
        msg += " [*]"       
    if i%5==0 or is_best:
        plt.hist(x=x[idxr], bins='auto', color='#ff6347',
                            alpha=0.7, rwidth=0.85)
        plt.hist(x=x[idxb], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85) 
        plt.grid(axis='y', alpha=0.75)
        plt.savefig("images/SiaRes/epoch_{}_{}_{}.png".format(i,lr,wd))
    print(msg.format(i, train_loss,train_acc, valid_loss,valid_acc))

    # check for improvement
    if not is_best:
        counter += 1
    if counter > train_patience:
        print("[!] No improvement in a while, stopping training.")
        break
    best_valid_loss = min(valid_loss, best_valid_loss)    
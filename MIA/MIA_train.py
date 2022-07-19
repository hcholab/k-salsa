# https://discuss.pytorch.org/t/target-size-torch-size-10-must-be-the-same-as-input-size-torch-size-2/72354/9
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import os
import sys
import random

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 

from torch.utils.data import Dataset,DataLoader
import PIL.Image as Image
import pandas as pd
from collections import Counter
import torchvision.transforms as transforms
import pdb
import einops
import glob

class dataset(Dataset):
    def __init__(self,df,data_path,image_transform=None): 
        super(Dataset,self).__init__()
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self,index):
        image_id = self.df['id_code'][index]
        image = Image.open(f'{self.data_path}/{image_id}.png')
        if self.image_transform :
            image = self.image_transform(image)
        
        label = self.df['centers'][index] 
        return image,label 

parser = argparse.ArgumentParser(description='MIA Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--epochs', default=50, type=int, help='learning rate')
parser.add_argument('--samek', default=5, type=int, help='learning rate')
parser.add_argument('--mult_batch', default=5, type=int, help='learning rate')
parser.add_argument('--train_df_path', default='../k-SALSA_algorithm/ECCV/aptos/labels/', type=str)
parser.add_argument('--train_image_path', default='./data/aptos_train', type=str)
parser.add_argument('--centroid_image_path', default='../k-SALSA_algorithm/aptos_k5_ours/', type=str)
parser.add_argument('--centroid_type', default='centroid_ours')
parser.add_argument('--data', default='aptos')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
mean = [0.41809403896331787, 0.22021695971488953, 0.06570685654878616]
std = [0.2725280523300171, 0.1409672647714615, 0.06416250765323639]

normalize = transforms.Normalize(mean=mean, std=std)

# K=5
train_df = pd.read_csv(args.train_df_path +'k'+ str(args.samek) +'_nearest_df.csv')

###############
# get centroid img by cluster id
###############
print(args.centroid_type)
print('centroid_image_path:',args.centroid_image_path)
dirs = glob.glob(args.centroid_image_path+'*')
dirs = sorted(dirs)
cid2iid = {}
for dir_ in dirs:
    img_id = dir_.split('/')[-1].split('.')[0]
    cluster_id = train_df[train_df['id_code'] == img_id]['centers'].values[0]
    cid2iid[cluster_id] = img_id

# pdb.set_trace()
train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

fname_lst = []
centers_lst = []
fnames = train_df.set_index(['id_code']).groupby(['centers']).groups
for i in range(len(fnames)):
    i=i+1
    fname_lst.append(fnames[i][0])
    centers_lst.append(i)

train_dataset = dataset(train_df, args.train_image_path, args, image_transform=train_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=int(args.samek * args.mult_batch),
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

# Model
print('==> Building model..')
net = resnet18(pretrained=True)

net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
net.fc = nn.Linear(512, 2)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, centers) in enumerate(train_loader):

        images, centers = images.to(device), centers.to(device)
        centroid, centroid_centers = images[0], centers[0]
        neg_imgs, neg_labels = images[1:], centers[1:]

        a = int(centroid_centers)
        image_id = train_df[train_df['centers'] == a]['id_code']
        image_pos_id = list(image_id)
        image_lst = []
        
        for i in range(len(image_pos_id)):
            image_pos = Image.open(args.train_image_path+f'/{image_pos_id[i]}.png')
            image_pos = train_transform(image_pos)
            image_lst.append(image_pos)

        pos_imgs = torch.stack(image_lst, dim=0).to(device)
        pos_labels = centroid_centers.repeat(len(pos_imgs))

        # get centroid img    
        centroid_id = cid2iid[a]

        if args.centroid_type =='centroid':
            centroid_id = centroid_id + '_centroids.png' 
        elif args.centroid_type == 'centroid_ours':
            centroid_id = centroid_id + '.png' 

        centroid = Image.open(args.centroid_image_path+f'/{centroid_id}')
        centroid = train_transform(centroid).to(device)

        pos_labels = centroid_centers.repeat(len(pos_imgs))

        input_data = torch.cat([pos_imgs, neg_imgs], dim=0) 
        input_labels = torch.cat([pos_labels, neg_labels])

        centroids = einops.repeat(centroid, 'c h w -> b c h w', b=len(input_data))

        input_data = torch.cat([input_data, centroids], dim=1)
        outputs = net(input_data)

        centroid_labels = centroid_centers.repeat(len(input_data))
        targets = torch.eq(centroid_labels, input_labels)
        targets = targets.type(torch.long)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

for epoch in range(args.epochs):
    train(epoch)
    scheduler.step()
    print('Saving..')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    if (epoch+1)%10 == 0:
        torch.save(net.state_dict(), './checkpoint/'+args.centroid_type + '_'+ args.data +'_'+ 'k' + str(args.samek) +'_'+str(epoch)+ '.pth')


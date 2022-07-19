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

class dataset(Dataset): # Inherits from the Dataset class.
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class. 
    
    Attributes :
        df:  DataFrame object for the csv file.
        data_path: Location of the dataset.
        image_transform: Transformations to apply to the image.
    '''
    
    def __init__(self,df,data_path,image_transform=None): # Constructor.
        super(Dataset,self).__init__() #Calls the constructor of the Dataset class.
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.df) #Returns the number of samples in the dataset.
    
    def __getitem__(self,index):
        # print('index:',index)
        # print('self.df[id_code]:',self.df['id_code'])
        image_id = self.df['id_code'][index]
        image = Image.open(f'{self.data_path}/{image_id}.png') #Image.
        if self.image_transform :
            image = self.image_transform(image) #Applies transformation to the image.
        
        label = self.df['centers'][index] #Label.
        return image,label #If train == True, return image & label.

parser = argparse.ArgumentParser(description='PyTorch aptos Training')
parser.add_argument('--samek', default=5, type=int, help='learning rate')
parser.add_argument('--ckpt_epochs', default=49, type=int, help='learning rate')
parser.add_argument('--topk', default=5, type=int, help='learning rate')
# parser.add_argument('--ckpt_path', default='/home/minkyu/privacy/ECCV2022/nearest_neighbor_samesize/ECCV/aptos/nearest_neighbor_centroid/labels/k5_aptos_nearest_df.csv', type=str, help='learning rate')
parser.add_argument('--centroid_type', default='centroid_ours')
parser.add_argument('--test_df_path', default='../k-SALSA_algorithm/ECCV/aptos/labels/', type=str)
parser.add_argument('--ckpt_path', default='./checkpoint/centroid_ours_aptos_k5_49.pth', type=str)
parser.add_argument('--test_image_path', default='./data/aptos_val', type=str)
parser.add_argument('--centroid_image_path', default='../k-SALSA_algorithm/aptos_k5_ours/', type=str)
parser.add_argument('--data', default='aptos')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('topk:',args.topk)
# Data
mean = [0.41809403896331787, 0.22021695971488953, 0.06570685654878616]
std = [0.2725280523300171, 0.1409672647714615, 0.06416250765323639]

normalize = transforms.Normalize(mean=mean, std=std)

# test_df = pd.read_csv('/home/unix/mjeon/privacy/ECCV2022/restyle-same-size-kmeans/ECCV/aptos/nearest_neighbor_centroid/labels/k'+str(args.samek)+'_full_test_aptos_nearest_df.csv')
test_df_path = args.test_df_path + '../k-SALSA_algorithm/save/aptos/labels/TEST_k'+str(args.samek)+'_' + args.data +'_nearest_df.csv' 
test_df = pd.read_csv(test_df_path)

centroid_lst = os.listdir(args.centroid_image_path)
print('K:',args.samek)
print(len(centroid_lst))

# load half data of full validation dataset
test1_df = test_df.sort_values(by="centers", ascending=True).groupby('centers').head().iloc[:len(test_df)//2].reset_index(drop=True)

fname_lst = []
centers_lst = []
fnames = test1_df.set_index(['id_code']).groupby(['centers']).groups
for i in range(len(fnames)):    
    for j, name in enumerate(fnames[i+1] + '.png'):
        if name in centroid_lst:
            fname_lst.append(fnames[i+1][j])
print(len(set(fname_lst)))
centers_lst = list(range(1,len(fname_lst)+1))

# only for test1 centroid
test1_df_only_centroid = {'id_code': fname_lst,
                         'centers':centers_lst}
test1_df_only_centroid = pd.DataFrame(test1_df_only_centroid, columns=['id_code', 'centers'])

train_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(), normalize])

Test1 = dataset(test1_df_only_centroid, args.centroid_image_path, image_transform=train_transform)
Test2 = dataset(test_df, args.test_image_path, image_transform=train_transform)

print('Test1:',len(Test1))
print('Test2:',len(Test2))

Test1_loader = DataLoader(
    Test1,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

Test2_loader = DataLoader(
    Test2,
    batch_size=len(test_df),
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

net = resnet18(pretrained=False)
net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
net.fc = nn.Linear(512, 2)
net = net.to(device)

print('args.centroid_type:',args.centroid_type)
# ckpt_path = './checkpoint/'+ args.centroid_type + '_' + args.data +'_k'+str(args.samek)+'_'+str(args.ckpt_epochs)+'.pth'
print('ckpt_path:',args.ckpt_path)
state_dict = torch.load(args.ckpt_path)

net.load_state_dict(state_dict)

train_loss = 0
correct = 0
total = 0
net.eval()
softmax = nn.Softmax(dim=-1)

score_matrices = torch.zeros(len(Test1_loader), len(Test2), 2).to(device)
print('score_matrices:',score_matrices.shape)
with torch.no_grad():
    for batch_idx, (centroid_img, centroid_centers) in enumerate(Test1_loader):
        centroid_img, centroid_centers = centroid_img.to(device), centroid_centers.to(device)
        Acc = []
        centroid = centroid_img[0]
        for idx, (imgs, labels) in enumerate(Test2_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            centroids = einops.repeat(centroid, 'c h w -> b c h w', b=len(imgs))
            input_data = torch.cat([imgs, centroids], dim=1)

            outputs = net(input_data)
            centroid_labels = centroid_centers.repeat(len(input_data))

            targets = torch.eq(centroid_labels, labels)
            targets = targets.type(torch.long)

            softmax_out = softmax(outputs)

            score_matrices[batch_idx] = softmax_out
            y_true = np.asarray(targets.cpu())
            y_score = np.asarray(softmax_out.cpu())

            ranked = np.argsort(y_score[:,1])
            n = len(y_score)
            descending_idx = ranked[::-1][:n]
            
            y_true_sorted = y_true[descending_idx]
            y_score_sorted = y_score[descending_idx]
            y_score_torch = torch.from_numpy(y_score_sorted)

            y_true_torch = torch.from_numpy(y_true_sorted)
            _, predicted2 = y_score_torch.max(1)

            topk_predicted = predicted2[:args.topk]
            topk_y_true = y_true_torch[:args.topk]

            total += args.topk
            correct += y_true_sorted[:args.topk].sum().item()
            topk_acc = 100.*correct/total
            progress_bar(idx, len(Test2_loader), 'Acc: %.3f%% (%d/%d)'
                            % (100.*correct/total, correct, total))
        Acc.append(topk_acc)

    print('done')
    print('Acc for all centroid:', np.asarray(Acc).mean())
    score_matrices = score_matrices.cpu().numpy()
    np.save(args.centroid_type+'k'+str(args.samek)+ '_' +args.data +'_val_score_matrices'+ str(args.ckpt_epochs) +'.npy', score_matrices)
    new_score_matrices = np.load(args.centroid_type+'k'+str(args.samek)+ '_' +args.data +'_val_score_matrices'+ str(args.ckpt_epochs) +'.npy')
    print('new_score_matrices:',new_score_matrices.shape)
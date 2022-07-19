import os
import sys
import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils.func import *
from train_baseline import train_2cls, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from data.transforms import data_transforms, simple_transform
from modules.builder import generate_model
from torch.utils.data import Dataset,DataLoader
import PIL.Image as Image
import pandas as pd
from collections import Counter
import wandb
import glob
import pdb

class retina_inverse_Dataset(Dataset):
    def __init__(self, k, df, data_path,image_transform=None):
        
        if k=='k2':
            print('k2')
            n=2500
        elif k=='k5':
            print('k5')
            n=1000
        elif k=='k10':
            print('k10')
            n=500
            
        self.df = df
        self.new_df = df.sample(n=n)
        self.new_df = self.new_df.reset_index(drop=True) #### label mapping 되는지 확인
        self.image_transform = image_transform
        self.data_path = data_path
        
    def __len__(self):
        return len(self.new_df)
    
    def __getitem__(self, index):
#         print('index:',index)
        img_path = os.path.join(self.data_path, self.new_df['id_code'][index] +".jpeg")
        img = Image.open(img_path)
        
        if(self.image_transform):
            img = self.image_transform(img)
        return img, torch.tensor(self.new_df['level'][index])
        
class retina_Dataset(Dataset):
    def __init__(self, df, data_path,image_transform):
        self.df = df    
        self.image_transform = image_transform
        self.data_path = data_path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
#         print('index:',index)
        img_path = os.path.join(self.data_path, self.df['image'][index] +".jpeg")
        img = Image.open(img_path)
        
        if(self.image_transform):
            img = self.image_transform(img)
        return img, torch.tensor(self.df['level'][index])

def make_labels(df, group_by_centers, data_name='eyepacs'):
    labels_lst = []
    fname_lst = []
    for i in range(len(group_by_centers)):
        i=i+1 # i+1 주의!!!
        fname = np.asarray((group_by_centers['image']))[i-1][1].iloc[0]
        fname_lst.append(fname)
        a = df[df['centers']==i]
        if data_name == 'eyepacs':
            a_array = np.asarray(a['level'])
        else:
            a_array = np.asarray(a['diagnosis'])
        cluster_label = Counter(a_array)
        labels = np.zeros([5])
        
        for j in range(5):
            labels[j] = cluster_label[j]
        labels_lst.append(labels) # 해당 class위치에 갯수
        
    labels_npy = np.asarray(labels_lst)
    labels_tensor = torch.from_numpy(labels_npy)
#     print('labels_tensor:',labels_tensor)
    normalized_labels = F.softmax(labels_tensor)
#     print('normalized_labels:',normalized_labels)

    normalized_labels_npy = np.asarray(normalized_labels)
    return normalized_labels_npy, fname_lst       

def epi(trial):
    args = parse_config()
    cfg = load_config(args.config)

    wandb.init(project="eyepacs_cls_org_v0", entity="eccv2022_")
    # create folder
    img_type = args.k  
    
    # create folder
    cfg.base.save_path = cfg.base.save_path  + '_' + img_type + '_' + str(trial) #+ 'inverse'
    save_path = cfg.base.save_path
    cfg.base.log_path = cfg.base.log_path  + '_' + img_type + '_' + str(trial) #+ 'inverse'
    log_path = cfg.base.log_path    
    
    if os.path.exists(save_path):
        warning = 'Save path {} exists.\nDo you want to overwrite it? (y/n)\n'.format(save_path)
        if not (args.overwrite or input(warning) == 'y'):
            sys.exit(0)
    else:
        os.makedirs(save_path)

    logger = SummaryWriter(log_path)
    copy_config(args.config, save_path)

    # print configuration
    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))

    # train
    # set_random_seed(cfg.base.random_seed)
    model = generate_model(cfg)

    # cfg.data.mean = mean
    # cfg.data.std = std

    train_transform, test_transform = data_transforms(cfg)
    # import pdb
    # pdb.set_trace()
    images_path = '/hub_data/privacy/ECCV/data/EyePACs/train_val_eyepacs/train' 

    eyepacs_label_csv = pd.read_csv('/hub_data/privacy/ECCV/data/EyePACs/train_val_eyepacs/labels/train_splited_eyepacs.csv')#'/hub_data/privacy/ECCV/data/splited_val/eyepacs/eyepacs_val_10000_2.csv')
    eyepacs_label_csv = eyepacs_label_csv.rename(columns={"image":"id_code", 'level':'level'})
    print(f'No.of.training_samples: {len(eyepacs_label_csv)}')

#     print([eyepacs_label_csv['level'].tolist().count(i)/10000 for i in range(5)])
#     pdb.set_trace()
    
    train_dataset = retina_inverse_Dataset(args.k, eyepacs_label_csv,images_path,image_transform=train_transform)
    print('len(train_dataset):',len(train_dataset))
    test_df = pd.read_csv('/hub_data/privacy/ECCV/data/splited_val/eyepacs/val1000_splited.csv')
    test_image_path = '/hub_data/privacy/ECCV/data/splited_val/eyepacs/sample1000_val'
    
    val_dataset = retina_Dataset(test_df, test_image_path, image_transform=test_transform)
    test_dataset = retina_Dataset(test_df, test_image_path, image_transform=test_transform)

    estimator = Estimator(cfg.train.criterion, cfg.data.num_classes)
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(save_path, 'best_validation_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    test_acc, test_kappa, test_confusion_matrix = evaluate(cfg, model, checkpoint, test_dataset, estimator)
    return test_acc, test_kappa, test_confusion_matrix, args.k

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    acc_lst = []
    kappa_lst = []
    confusion_matrix_lst = []
    
    for trial in range(2):
        test_acc, test_kappa, test_confusion_matrix, k   = epi(trial)
        acc_lst.append(test_acc)
        kappa_lst.append(test_kappa)
        confusion_matrix_lst.append(test_confusion_matrix)
        
    print('acc: ', acc_lst)
    print('kappa: ', kappa_lst)
    print('confusion_matrix: ', confusion_matrix_lst)
    
    wandb.log({"acc": np.mean(acc_lst),
               "kappa": np.mean(kappa_lst), 
               "k": k})
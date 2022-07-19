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
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from data.transforms import data_transforms, simple_transform
from modules.builder import generate_model
from torch.utils.data import Dataset,DataLoader
import PIL.Image as Image
import pandas as pd
from collections import Counter
import wandb
import argparse
import pdb

class aptos_centroid_dataset(Dataset):
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class. 
    
    Attributes :
        data_path: Location of the dataset.
        image_transform: Transformations to apply to the image.
        train: A boolean indicating whether it is a training_set or not.
    '''
    
    def __init__(self,fname_lst,data_path,labels,image_transform=None, is_centroid=False): 
        super(Dataset,self).__init__()
        self.fname_lst = fname_lst
        self.data_path = data_path
        self.image_transform = image_transform
        self.labels = labels
        self.is_centroid = is_centroid
        
    def __len__(self):
        return len(self.fname_lst)
    
    def __getitem__(self,index):
        image_id = self.fname_lst[index]
        if self.is_centroid:
            image = Image.open(f'{self.data_path}/{image_id}_centroids.png') #Image.
        else:
            image = Image.open(f'{self.data_path}/{image_id}.png') 
        if self.image_transform :
            image = self.image_transform(image) 
        
        label = self.labels[index]

        return image,label 

            
class aptos_dataset(Dataset): 
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class. 
    
    Attributes :
        df:  DataFrame object for the csv file.
        data_path: Location of the dataset.
        image_transform: Transformations to apply to the image.
        train: A boolean indicating whether it is a training_set or not.
    '''
    
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
        
        label = self.df['diagnosis'][index] 
        return image,label 

def make_labels(df, group_by_centers, num_classes=5, data_name='aptos'):
    labels_lst = []
    fname_lst = []
    for i in range(len(group_by_centers)):
        i=i+1 
        fname = np.asarray((group_by_centers['id_code']))[i-1][1].iloc[0]
        fname_lst.append(fname)
        a = df[df['centers']==i]
        if data_name == 'eyepacs':
            a_array = np.asarray(a['level'])
        else:
            a_array = np.asarray(a['diagnosis'])
        cluster_label = Counter(a_array)
        labels = np.zeros([num_classes])
        
        for j in range(num_classes):
            labels[j] = cluster_label[j]
        labels_lst.append(labels) 
        
    labels_npy = np.asarray(labels_lst)
    labels_tensor = torch.from_numpy(labels_npy)
    normalized_labels = F.softmax(labels_tensor)

    normalized_labels_npy = np.asarray(normalized_labels)
    return normalized_labels_npy, fname_lst

        
def epi(trial):
    # environment
    args = parse_config()
    cfg = load_config(args.config)

    wandb.init(project="aptos_cls_k-SALSA", entity="eccv2022_")
    ################################################
    ############ CHECK
    ################################################
    img_type = args.k + '_' + args.crop + '_' + args.lambda1 + '_' + args.lambda2 
    
    # create folder
    cfg.base.save_path = cfg.base.save_path  + '_' + img_type + '_' + str(trial)
    save_path = cfg.base.save_path 
    cfg.base.log_path = cfg.base.log_path  + '_' + img_type + '_' + str(trial)
    log_path = cfg.base.log_path 
    ################################################
    ############ CHECK
    ################################################
    wandb.config = {
        "dataname": args.config,
        "k": args.k,
        "crop": args.crop,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2
    }

#     pdb.set_trace()
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
    model = generate_model(cfg)
    train_transform, test_transform = data_transforms(cfg)
    
    ################################################
    ############ CHECK
    ###############################################
    images_path = args.images_path #'/home/unix/mjeon/ECCV2022/privacy/ECCV2022/restyle-same-size-kmeans/Ablation_eccv_aptos_k5_crop4_1e2_1e-6_no_alignment/style_proj'

    # Make labels
    aptos_df = pd.read_csv(args.df)
    print(f'No.of.training_samples: {len(aptos_df)}')
    ################################################
    ############ CHECK
    ###############################################

    aptos_label_csv = pd.read_csv(args.labels_train)

    df_aptos_merge = pd.merge(aptos_df, aptos_label_csv, on='id_code')
    group_by_centers_aptos = df_aptos_merge.groupby(['centers'])
    aptos_normalized_labels_npy, fname_lst = make_labels(df_aptos_merge, group_by_centers_aptos, num_classes=5, data_name='aptos')
    label_dist = aptos_normalized_labels_npy
    max_label = torch.max(torch.tensor(label_dist),1)[1]
    print([max_label.tolist().count(i)/len(max_label) for i in range(5)])
    print('group_by_centers_aptos:',len(group_by_centers_aptos))
    print('fname_lst:', len(fname_lst))

    train_dataset = aptos_centroid_dataset(fname_lst,images_path,aptos_normalized_labels_npy,image_transform=train_transform, is_centroid=args.is_centroid)

    test_df = pd.read_csv(args.labels_test)
    test_image_path = args.test_images_path #'/home/unix/mjeon/data/splited_val/aptos/val'
    
    val_dataset = aptos_dataset(test_df, test_image_path, image_transform=test_transform)
    test_dataset = aptos_dataset(test_df, test_image_path, image_transform=test_transform)

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
    acc, kappa, confusion_matrix = evaluate(args, cfg, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    test_acc, test_kappa, test_confusion_matrix = evaluate(args, cfg, model, checkpoint, test_dataset, estimator)
    print('images_path:',images_path)
    return test_acc, test_kappa, test_confusion_matrix,args.k , args.crop, args.lambda1 , args.lambda2 
    
def main():
    
    acc_lst = []
    kappa_lst = []
    confusion_matrix_lst = []
    
    for trial in range(1):
        test_acc, test_kappa, test_confusion_matrix, k , crop, lambda1 , lambda2  = epi(trial)
        acc_lst.append(test_acc)
        kappa_lst.append(test_kappa)
        confusion_matrix_lst.append(test_confusion_matrix)
        
    print('acc: ', acc_lst)
    print('kappa: ', kappa_lst)
    print('confusion_matrix: ', confusion_matrix_lst)

    wandb.log({"acc": np.mean(acc_lst),
               "kappa": np.mean(kappa_lst), 
               "k": k,
               "crop": crop,
               "lambda1": lambda1,
               "lambda2": lambda2})
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()

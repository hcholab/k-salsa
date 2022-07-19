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

class aptos_centroid_dataset(Dataset): # Inherits from the Dataset class.
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class. 
    
    Attributes :
        df:  DataFrame object for the csv file.
        data_path: Location of the dataset.
        image_transform: Transformations to apply to the image.
        train: A boolean indicating whether it is a training_set or not.
    '''
    
    def __init__(self,df,fname_lst,data_path,labels,image_transform=None): # Constructor.
        super(Dataset,self).__init__() #Calls the constructor of the Dataset class.
        self.fname_lst = fname_lst
        self.data_path = data_path
        self.image_transform = image_transform
        self.labels = labels
        # self.normalized_labels = normalized_labels
        
    def __len__(self):
        return len(self.fname_lst) #Returns the number of samples in the dataset.
    
    def __getitem__(self,index):
        image_id = self.fname_lst[index]
        image = Image.open(f'{self.data_path}/{image_id}.jpeg') #Image.
        if self.image_transform :
            image = self.image_transform(image) #Applies transformation to the image.
        label = self.labels[index]

        # label = self.df['diagnosis'][index] #self.normalized_labels[index]
        return image,label #,image_id,index #If train == True, return image & label.

            
class aptos_dataset(Dataset): # Inherits from the Dataset class.
    '''
    dataset class overloads the __init__, __len__, __getitem__ methods of the Dataset class. 
    
    Attributes :
        df:  DataFrame object for the csv file.
        data_path: Location of the dataset.
        image_transform: Transformations to apply to the image.
        train: A boolean indicating whether it is a training_set or not.
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
        
        label = self.df['diagnosis'][index] #Label.
        return image,label #If train == True, return image & label.

def make_labels(df, group_by_centers, data_name='aptos'):
    labels_lst = []
    fname_lst = []
    for i in range(len(group_by_centers)):
        i=i+1 # i+1 주의!!!
        fname = np.asarray((group_by_centers['id_code']))[i-1][1].iloc[0]
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

def main():
    args = parse_config()
    cfg = load_config(args.config)

    # create folder
    save_path = cfg.base.save_path
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
    # train_df = pd.read_csv('/home/minkyu/privacy/ECCV2022/restyle-same-size-kmeans/ECCV/aptos/labels/localstyle_crop8_1e2_1e_6_yhat_aptos_k5_with_label.csv')
    # images_path = '/home/unix/mjeon/privacy/ICML2022/same_size/restyle-same-size-kmeans/ECCV/aptos/centroid_data/k5_images'
    images_path = '/home/minkyu/privacy/ECCV2022/nearest_neighbor_samesize/ECCV_aptos_k5_nearest_samesize/inference_results/4'

    # Make labels
    aptos_df = pd.read_csv('/home/minkyu/privacy/ECCV2022/nearest_neighbor_samesize/ECCV/aptos/nearest_neighbor_centroid/labels/k5_aptos_nearest_df.csv')
    

    aptos_label_csv = pd.read_csv('/hub_data/privacy/ECCV/data/splited_val/aptos/aptos_val_3000.csv')
    df_aptos_merge = pd.merge(aptos_df, aptos_label_csv, on='id_code')
    # df_k5_aptos_W16.to_csv('./labels/df_k5_aptos_w16.csv')
    group_by_centers_aptos = df_aptos_merge.groupby(['centers'])
    aptos_normalized_labels_npy, fname_lst = make_labels(df_aptos_merge, group_by_centers_aptos, data_name='aptos')
    print('group_by_centers_aptos:',len(group_by_centers_aptos))
    print('fname_lst:', len(fname_lst))

    # normalized_labels = np.load('/home/minkyu/privacy/ICML2022/classification/labels/aptos/whole_k5_aptos_W16_normalized_labels.npy')
    # train_dataset = aptos_centroid_dataset(train_df,images_path,normalized_labels,image_transform=train_transform)
    train_dataset = aptos_centroid_dataset(aptos_df,fname_lst,images_path,aptos_normalized_labels_npy,image_transform=train_transform)
    print(f'No.of.training_samples: {len(train_dataset)}')
    test_df = pd.read_csv('/hub_data/privacy/ECCV/data/splited_val/aptos/val_splited.csv')
    test_image_path = '/hub_data/privacy/ECCV/data/splited_val/aptos/aptos_val/val'
    
    val_dataset = aptos_dataset(test_df, test_image_path, image_transform=test_transform)
    test_dataset = aptos_dataset(test_df, test_image_path, image_transform=test_transform)

    estimator = Estimator(cfg.train.criterion, cfg.data.num_classes)
#     wandb.init(project="test-project", entity="eccv2022_")
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
    evaluate(cfg, model, checkpoint, test_dataset, estimator)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()

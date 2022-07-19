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
from train import train_2cls, evaluate
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

class retina_centroid_Dataset(Dataset): # Inherits from the Dataset class.
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
        image = Image.open(f'{self.data_path}/{image_id}._centroids.png') #Image.
        if self.image_transform :
            image = self.image_transform(image) #Applies transformation to the image.
        label = self.labels[index]

        # label = self.df['diagnosis'][index] #self.normalized_labels[index]
        return image,label #,image_id,index #If train == True, return image & label.

            
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

# def make_labels(df, group_by_centers, data_name='aptos'):
#     labels_lst = []
#     fname_lst = []
#     for i in range(len(group_by_centers)):
#         i=i+1 # i+1 주의!!! (centers가 1부터 시작해)
#         fname = np.asarray((group_by_centers['id_code']))[i-1][1].iloc[0]
#         fname_lst.append(fname)
#         a = df[df['centers']==i]
#         if data_name == 'eyepacs':
#             a_array = np.asarray(a['level'])centroid.
#         else:
#             a_array = np.asarray(a['diagnosis'])
#         cluster_label = Counter(a_array)
#         labels = np.zeros([5])
        
#         for j in range(5):
#             labels[j] = cluster_label[j]
#         labels_lst.append(labels) # 해당 class위치에 갯수
        
#     labels_npy = np.asarray(labels_lst)
#     labels_tensor = torch.from_numpy(labels_npy)
# #     print('labels_tensor:',labels_tensor)
#     normalized_labels = F.softmax(labels_tensor)
# #     print('normalized_labels:',normalized_labels)

#     normalized_labels_npy = np.asarray(normalized_labels)
#     return normalized_labels_npy, fname_lst

        
def epi():
    args = parse_config()
    cfg = load_config(args.config)

    wandb.init(project="eyepacs_cls_v5", entity="eccv2022_")
    ################################################
    ############ CHECK
    ###############################################
    img_type = args.k + '_' + args.crop + '_' + args.lambda1 + '_' + args.lambda2 
    
    # create folder
    cfg.base.save_path = cfg.base.save_path  + '_' + img_type + 'ours' +'tmp'
    save_path = cfg.base.save_path 
    cfg.base.log_path = cfg.base.log_path  + '_' + img_type + 'ours' +'tmp'
    log_path = cfg.base.log_path 
    ################################################
    ############ CHECK
    ###############################################
    wandb.config = {
        "dataname": args.config,
        "k": args.k,
        "crop": args.crop,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2
    }
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
    
    ################################################
    ############ CHECK
    ###############################################
    images_path = '/home/minkyu/privacy/ECCV2022/nearest_neighbor_samesize/new_eccv_nearest_results/NEW_eccv_nearest_eyepacs_'+ args.k +'_crop4_1e2_1e-6/centroid'

    # Make labels
    eyepacs_df = pd.read_csv('/home/minkyu/privacy/ECCV2022/nearest_neighbor_samesize/ECCV/eyepacs/nearest_neighbor_centroid/labels/NEW_'+args.k+'_eyepacs_nearest_df.csv')
    print(f'No.of.training_samples: {len(eyepacs_df)}')
    
    ################################################
    ############ CHECK
    ###############################################
    eyepacs_label_csv = pd.read_csv('/hub_data/privacy/ECCV/data/EyePACs/train_val_eyepacs/labels/train_splited_eyepacs.csv')
    #'/hub_data/privacy/ECCV/data/splited_val/eyepacs/eyepacs_val_10000_2.csv')
    eyepacs_label_csv = eyepacs_label_csv.rename(columns={"image":"id_code", 'level':'level'})
    df_eyepacs_merge = pd.merge(eyepacs_df, eyepacs_label_csv, on='id_code')
    # print('len(eyepacs_label_csv):',len(eyepacs_label_csv))
    #########################
    ##### update eyepacs ####
    #########################
    img_ids = glob.glob(images_path+'/*')
    img_ids = [img_id.split('/')[-1].split('.')[0] for img_id in img_ids]
    centers, eyepacs_normalized_labels_npy = [], []
    print('img_ids:',len(img_ids))
    for img_id in img_ids:
        img_idx = np.where(eyepacs_df['id_code']==img_id)[0][0]
        center = eyepacs_df.iloc[[img_idx]]['centers'].values[0]
        centers.append(center)
        c_ids = np.where(df_eyepacs_merge['centers']==center)[0] #[] 
        labels = [ df_eyepacs_merge['level'][c_id] for c_id in c_ids ]
        labels = [labels.count(i) for i in range(5)]
        division = np.sum(labels)
        eyepacs_normalized_labels_npy.append((labels/division).tolist())
    eyepacs_normalized_labels_npy = np.array(eyepacs_normalized_labels_npy)
    fname_lst = img_ids
    print('eyepacs_normalized_labels_npy:',len(eyepacs_normalized_labels_npy))
    
    label_dist = eyepacs_normalized_labels_npy
    max_label = torch.max(torch.tensor(label_dist),1)[1]
    print([max_label.tolist().count(i)/len(max_label) for i in range(5)])
#     import pdb
#     pdb.set_trace()
    try:
        label_dist = eyepacs_normalized_labels_npy
        max_label = torch.max(torch.tensor(label_dist),1)[1]
        [max_label.tolist().count(i)/len(max_label) for i in range(5)]
    except:
        import pdb
        # pdb.set_trace()
    
#     print('group_by_centers_eyepacs:',len(group_by_centers_eyepacs))
    print('fname_lst:', len(fname_lst)) # 0

    # normalized_labels = np.load('/home/minkyu/privacy/ICML2022/classification/labels/aptos/whole_k5_aptos_W16_normalized_labels.npy')
    # train_dataset = aptos_centroid_dataset(train_df,images_path,normalized_labels,image_transform=train_transform)
    train_dataset = retina_centroid_Dataset(eyepacs_df,fname_lst,images_path,eyepacs_normalized_labels_npy,image_transform=train_transform)
    print('train_dataset:',len(train_dataset))
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

    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(save_path, 'best_validation_weights.pt')
    acc, kappa, confusion_matrix = evaluate(cfg, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    test_acc, test_kappa, test_confusion_matrix = evaluate(cfg, model, checkpoint, test_dataset, estimator)
    
    return test_acc, test_kappa, test_confusion_matrix, acc, kappa, confusion_matrix, args.k , args.crop, args.lambda1 , args.lambda2

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    
#     acc_lst = []
#     kappa_lst = []
#     confusion_matrix_lst = []
#     for _ in range(2):
    test_acc, test_kappa, test_confusion_matrix, acc, kappa, confusion_matrix,k , crop, lambda1 , lambda2 = epi()
#         acc_lst.append(test_acc)
#         kappa_lst.append(test_kappa)
#         confusion_matrix_lst.append(test_confusion_matrix)
        
    print('acc: ', test_acc)
    print('kappa: ', test_kappa)
    print('confusion_matrix: ', test_confusion_matrix)
    
    wandb.log({"best_acc": acc,
               "best_kappa":kappa,
               "final_acc": test_acc,
               "final_kappa": test_kappa, 
               "k": k,
               "crop": crop,
               "lambda1": lambda1,
               "lambda2": lambda2})
    
if __name__ == '__main__':
    main()

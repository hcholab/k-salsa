import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import torch.nn as nn

sys.path.append(".")
sys.path.append("..")
import pandas as pd
from configs import data_configs
from datasets.inference_dataset import Clustering_Dataset
from options.test_options import TestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, get_average_image
import random
from collections import Counter
from numpy import save
from numpy import load
import torch.nn.functional as F
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt
import scipy.spatial.distance

# torch.manual_seed(2021)
# random.seed(2021)
# np.random.seed(2021)

# takes N by N precomputed distance matrix D
# outputs a length N cluster assignment vector with indicies 1..C for C clusters
def nearestNeighborClust(D, clust_size=2):
    D = D.copy()
    n = D.shape[0]
    assert D.shape[1] == n

    D += np.diag([np.nan] * n)
    def maxD(M, allowed):
        s = np.nansum(M,axis=1)
        s[~allowed] = np.nan
        return np.nanargmax(s)
    
    clust_inds = np.zeros(n, dtype=np.int) + np.nan
    cur_index = 1
    while np.any(np.isnan(clust_inds)):
        # Choose maxD point
        next_point = maxD(D, np.isnan(clust_inds))
    
        # Find clust_size-1 closest points
        nearest = np.argsort(D[next_point,:])[:clust_size-1]
        
        # Assign cluster
        cluster = np.append(nearest, next_point).astype(np.int)
        clust_inds[cluster] = cur_index
        
        # Update matrix
        D[cluster,:] = np.nan
        D[:,cluster] = np.nan
        
        cur_index += 1
    return clust_inds.astype(np.int)

def compute_features(eval_loader, model, avg_image, opts):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),512*16).cuda()
    for i, (images, index, from_path) in enumerate(eval_loader):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            _, result_batch, result_latents, feat = run_on_batch(images, model, opts, avg_image, computing_features=True)
            feat = feat.view(feat.shape[0],-1)
            features[index] = feat
    return features.cpu()


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net = pSp(opts)
    else:
        net = e4e(opts)

    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    print('data_path:',opts.data_path)
    train_df = pd.read_csv(opts.df)
    dataset = Clustering_Dataset(df=train_df, data_path = opts.data_path,
                                        transform=transforms_dict['transform_inference'],
                                        opts=opts)

    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            pin_memory=True,
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, opts)

    print('opts.same_k:',opts.same_k)
    features = compute_features(dataloader, net, avg_image, opts)
    features = features.numpy()
    # Model Save and Load
    save('./ECCV/'+opts.data+'/features/k5_features.npy', features)
    # features = load('./ECCV/'+ opts.data +'/features/k'+ opts.same_k +'_features.npy')
    
    predistance = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(features))
    centroid_idx = nearestNeighborClust(predistance, clust_size=opts.same_k)
    # Model Save and Load
    # save('./ECCV/'+opts.data+'/labels/k'+str(opts.same_k)+'_nearest_centroids_idx.npy', centroid_idx)
    # centroid_idx = load('./ECCV/'+opts.data+'/labels/k'+str(opts.same_k)+'_nearest_centroids_idx.npy')

    counts = dict()
    for i in centroid_idx:
        counts[i] = counts.get(i, 0) + 1
        
    num_count_centers_lst = []
    for i in range(len(centroid_idx)):
        add = centroid_idx[i]
        num_count_centers_lst.append(counts[add])
    
    if opts.data == 'aptos':
        id_code = train_df['id_code']
    elif opts.data == 'eyepacs':
        id_code = train_df['image']

    new_data = {'id_code': id_code,
          'centers': centroid_idx,
           'count_centers':np.array(num_count_centers_lst)}
    
    new_df = pd.DataFrame(new_data, columns=['id_code','centers','count_centers']) 
    new_df.to_csv('./ECCV/'+opts.data+'/labels/k'+str(opts.same_k)+'_nearest_df.csv', index=False)
    a = np.unique(centroid_idx, return_counts=True)
    each_clusters = a[1]
    count_cluster = Counter(each_clusters)
    print('count_cluster:',count_cluster)

if __name__ == '__main__':
    run()

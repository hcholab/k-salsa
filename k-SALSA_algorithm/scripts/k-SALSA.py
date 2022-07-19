import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import sys
import torchvision.transforms as T
import numpy as np

sys.path.append(".")
sys.path.append("..")
from criteria import id_loss, w_norm, moco_loss

import os
from argparse import Namespace
from tqdm import tqdm
import time
from torch.utils.data import DataLoader

from configs import data_configs
from datasets.inference_dataset import InferenceDataset_for_centroid, Clustering_Dataset
from options.test_options import TestOptions
from models.psp import pSp
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, get_average_image
from collections import Counter
from numpy import save
from numpy import load

import pandas as pd
from torchvision.models import vgg19
from torchvision import models
import torch.nn as nn

import matplotlib.pyplot as plt
import scipy.spatial.distance

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

class Vgg19(nn.Module):
    def __init__(self, device):
        super(Vgg19, self).__init__()
        self.vgg19 = vgg19(pretrained=True).requires_grad_(False).eval().to(device)
        
    def forward(self, x):
        features = []
        for layer in self.vgg19.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features

def gram_matrix(features):
    # first dimension is batch dimension
    _, c, h, w = features.size()
    features = features.reshape(c, h*w)
    return features@features.T

def project(
    net,
    moco_l,
    opts,
    avg_image,
    source_images: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    def logprint(*args):
        if verbose:
            print(*args)

    y_hat, latents, _, _, _ = run_on_batch(source_images, net, opts, avg_image, computing_features=True, get_latent=True)
    w_avg = torch.mean(latents, dim=0, keepdim=True)

    source_images_content = y_hat
    model = Vgg19(device=device)
    source_images2 = net.face_pool(source_images)
    source_features = model(source_images2)

    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    
    style_layers=1 
    cos = nn.CosineSimilarity(dim=1)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    for step in range(opts.num_steps):
        # Learning rate schedule.
        t = step / opts.num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        y_hat_synth,_ = net.decoder([w_opt],
                                        input_is_latent=True,
                                        randomize_noise=False,
                                        return_latents=True)
        if step==0:
            y_hat_centroid_origin = y_hat_synth
        
        # < Content >
        y_hat_synth = net.face_pool(y_hat_synth) # [1,3,256,256]
        y_hat_synth2 = y_hat_synth.repeat([source_images.shape[0],1,1,1]) # [1,3,256,256] -> [k,3,256,256]
        
        # <Content Loss >
        loss_moco, sim_improvement, id_logs = moco_l(y_hat_synth2, source_images_content, source_images_content)

        # < Style Loss crop>
        centroid_features = model(y_hat_synth)
        style_loss = 0
        channel_size = centroid_features[style_layers].size(1) # 64
        centroid_gram = torch.zeros(opts.num_crop*opts.num_crop, channel_size, channel_size).to(device)
        target_gram = torch.zeros(len(source_images), opts.num_crop*opts.num_crop, channel_size, channel_size).to(device)
        
        for idx in range(len(source_images)):
            t=0
            target_feature = torch.unsqueeze(source_features[style_layers][idx],0) # [1,64,256,256]
            feature_size = target_feature.size(2) # 256
            crop_size = feature_size // opts.num_crop # 256//4

            for i in range(opts.num_crop):
                for j in range(opts.num_crop):
                    new_target_feat = T.functional.crop(target_feature, top=i*crop_size, left=j*crop_size, height=crop_size, width=crop_size) # [1,64,64,64]
                    new_target_style = gram_matrix(new_target_feat) # [64,64]
                    target_gram[idx][t] = new_target_style
                    t = t+1
        
        u=0
        for i in range(opts.num_crop):
            for j in range(opts.num_crop):
                new_centroid_feat = T.functional.crop(centroid_features[style_layers], top=i*crop_size, left=j*crop_size, height=crop_size, width=crop_size)
                new_centroid_style = gram_matrix(new_centroid_feat)
                centroid_gram[u] = new_centroid_style
                u=u+1

        centroid_gram = centroid_gram.view(centroid_gram.size(0), -1) # for centroid 
        target_gram = target_gram.view(target_gram.size(0), target_gram.size(1), -1) # for similar image(target)

        # Get alignment patches
        style_losses = []
        for idx in range(len(source_images)):
            pos_idxes = []
            style_loss = 0
            for i in range(target_gram.size(1)):
                cos_output = cos(centroid_gram, torch.unsqueeze(target_gram[idx][i],0)) # 1 target, whole centroid
                pos_idx = cos_output.argmax(dim=-1)
                pos_idxes.append(pos_idx)
            for ix in range(len(pos_idxes)):
                style_loss += ((centroid_gram[pos_idxes[ix]] - target_gram[idx][ix].detach())**2).sum() 
            style_losses.append(style_loss/len(pos_idxes))
        local_style_loss = sum(style_losses)/len(style_losses)
        loss = opts.content_weight*loss_moco + opts.style_weight*local_style_loss 

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{opts.num_steps}: loss_moco {float(opts.content_weight*loss_moco):<5.2f}, style_loss {float(opts.style_weight*local_style_loss):<5.2f}')
    return w_opt.detach(), y_hat_centroid_origin.detach()

def same_size_cluster(net, avg_image, dataset_args, transforms_dict, opts):
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

    features = compute_features(dataloader, net, avg_image, opts)
    features = features.numpy()
    # Model Save and Load
    save('./save/'+opts.data+'/features/k' +str(opts.same_k) +'_features.npy', features)
    # features = load('./save/'+opts.data+'/features/k' +str(opts.same_k) +'_features.npy')
    
    predistance = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(features))
    centroid_idx = nearestNeighborClust(predistance, clust_size=opts.same_k)
    # Model Save and Load
    # save('./save/'+opts.data+'/labels/k'+str(opts.same_k)+'_nearest_centroids_idx.npy', centroid_idx)
    # centroid_idx = load('./save/'+opts.data+'/labels/k'+str(opts.same_k)+'_nearest_centroids_idx.npy')

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
    new_df.to_csv('./save/'+opts.data+'/labels/k'+str(opts.same_k)+'_nearest_df.csv', index=False)
    a = np.unique(centroid_idx, return_counts=True)
    each_clusters = a[1]
    count_cluster = Counter(each_clusters)
    print('count_cluster:',count_cluster)

def run_projection():
    seed = 2022
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    test_opts = TestOptions().parse()

    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    save_dir_centroid = os.path.join(test_opts.exp_dir, 'centroid')
    save_dir_proj = os.path.join(test_opts.exp_dir, 'style_proj')

    net = pSp(opts)
    net.eval()
    net.cuda()
    
    assert opts.test_batch_size == opts.same_k

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, opts)

    # Same Size Clustering
    same_size_cluster(net, avg_image, dataset_args, transforms_dict, opts)

    labels = pd.read_csv('./save/'+opts.data+'/labels/k'+str(opts.same_k)+'_nearest_df.csv')
    dataset = InferenceDataset_for_centroid(root=opts.data_path,
                            transform=transforms_dict['transform_inference'],
                            opts=opts,
                            labels=labels)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)


    # Load networks.
    device = torch.device('cuda')
    
    net = copy.deepcopy(net).eval().requires_grad_(False).to(device)
    moco_l = moco_loss.MocoLoss()

    # Optimize projection.
    start_time = perf_counter()
    global_i = 0
    global_time = []
    all_latents = {}
    for (input_batch, index, from_path) in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            source_image = input_batch.cuda().float() # [B,3,256,256]

        projected_w, centroid_origin = project(
            net=net,
            moco_l=moco_l,
            opts=opts,
            avg_image=avg_image,
            source_images=source_image,
            device=device,
            verbose=True
        )
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
        synth_image,_ = net.decoder([projected_w],
                                            input_is_latent=True,
                                            randomize_noise=False,
                                            return_latents=True)
        
        synth_batches = {idx: [] for idx in range(synth_image.shape[0])}
        centroid_origin_batches = {idx: [] for idx in range(centroid_origin.shape[0])}
        synth_batches[0].append(synth_image[0])
        centroid_origin_batches[0].append(centroid_origin[0])

        centroid_image = tensor2im(centroid_origin[0]) 
        os.makedirs(save_dir_proj, exist_ok=True)
        os.makedirs(save_dir_centroid, exist_ok=True)

        for i in range(synth_image.shape[0]):
            im_path = dataset.lst[global_i]
            # Project
            results_project = tensor2im(synth_batches[i][0])
            results_project.resize(resize_amount).save(os.path.join(save_dir_proj, os.path.basename(im_path)))
            centroid_image.resize(resize_amount).save(os.path.join(save_dir_centroid, os.path.basename(im_path)[:-4]+'_centroids.png'))
            
            global_i += input_batch.shape[0]
#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

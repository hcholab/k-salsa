import subprocess
import argparse
from tqdm import tqdm
import pdb
import glob

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# cuda = '2'

data_names = ['eyepacs', 'aptos']
ks = ['k2', 'k10', 'k5']
img_types = ['pixel', 'pca']

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_name', type=str, required=True, help=('Path to the true images'))
parser.add_argument('--cuda', type=str, required=True, help=('Path to the true images'))

args = parser.parse_args()
 
for k in ks:
    for img_type in img_types:
        if args.data_name =='aptos':
            subprocess.call(f"CUDA_VISIBLE_DEVICES={args.cuda} python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type {img_type} --k {k} --gpu {args.cuda}", shell=True)
#             subprocess.call(f"CUDA_VISIBLE_DEVICES={args.cuda} python kid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type {img_type} --k {k} --gpu {args.cuda}", shell=True)
            
        elif args.data_name =='eyepacs':
            subprocess.call(f"CUDA_VISIBLE_DEVICES={args.cuda} python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type {img_type} --k {k} --gpu {args.cuda}", shell=True)
#             subprocess.call(f"CUDA_VISIBLE_DEVICES={args.cuda} python kid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type {img_type} --k {k} --gpu {args.cuda}", shell=True)


# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type pixel --k k2 --gpu 6
# CUDA_VISIBLE_DEVICES=2 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type pixel --k k5 --gpu 2
# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type pixel --k k10 --gpu 6

# # CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type pca --k k2 --gpu 6
# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type pca --k k5 --gpu 6
# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name aptos --img_type pca --k k10 --gpu 6




# # CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type pixel --k k2 --gpu 6
# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type pixel --k k5 --gpu 6
# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type pixel --k k10 --gpu 6

# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type pca --k k2 --gpu 6
# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type pca --k k5 --gpu 6
# CUDA_VISIBLE_DEVICES=6 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type pca --k k10 --gpu 6





# CUDA_VISIBLE_DEVICES=2 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type ours --k k2 --gpu 2
# CUDA_VISIBLE_DEVICES=3 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type ours --k k5 --gpu 3
# CUDA_VISIBLE_DEVICES=4 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type ours --k k10 --gpu 4

# CUDA_VISIBLE_DEVICES=2 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type centroid --k k2 --gpu 2
# CUDA_VISIBLE_DEVICES=3 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type centroid --k k5 --gpu 3
# CUDA_VISIBLE_DEVICES=4 python fid_score.py --true ~/privacy/ECCV2022/preprocess/ --fake ~/privacy/ECCV2022/preprocess/ --data_name eyepacs --img_type centroid --k k10 --gpu 4



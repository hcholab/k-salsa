import subprocess
import argparse
from tqdm import tqdm
import pdb
import glob


cuda = '1'#['0'][args.n % 2]

dirs = glob.glob('/home/minkyu/privacy/ECCV2022/nearest_neighbor_samesize/eccv_nearest_aptos_*')
dirs = sorted(dirs)

for dir_ in dirs:
    dir_ = dir_.split('/')[-1]
    k, crop, lambda1, lambda2 = dir_.split('_')[-4:]
    subprocess.call(f"CUDA_VISIBLE_DEVICES={cuda} python main_centroid.py --k {k} --crop {crop} --lambda1 {lambda1} --lambda2 {lambda2} -config configs/aptos.yaml", shell=True)

# CUDA_VISIBLE_DEVICES=5 python main_centroid.py --k k5 --crop crop8 --lambda1 1e2 --lambda2 1e-6 -config configs/aptos.yaml
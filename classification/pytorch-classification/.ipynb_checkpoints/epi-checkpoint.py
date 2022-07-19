import subprocess
import argparse
from tqdm import tqdm
import pdb
import glob


cuda = '0'#['0'][args.n % 2]

dirs = glob.glob('/home/minkyu/privacy/ECCV2022/nearest_neighbor_samesize/eccv_nearest_aptos_*')
dirs = sorted(dirs)

for dir_ in dirs:
    dir_ = dir_.split('/')[-1]
    k, crop, lambda1, lambda2 = dir_.split('_')[-4:]
    subprocess.call(f"CUDA_VISIBLE_DEVICES={cuda} python main_ours.py --k {k} --crop {crop} --lambda1 {lambda1} --lambda2 {lambda2} -config configs/aptos.yaml", shell=True)


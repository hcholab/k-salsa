import yaml
import torch
import shutil
import argparse

from tqdm import tqdm
from munch import munchify
from torch.utils.data import DataLoader

from utils.const import regression_loss


def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '-config',
        type=str,
        default='./configs/default.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-overwrite',
        action='store_true',
        default=False,
        help='Overwrite file in the save path.'
    )
    parser.add_argument(
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    parser.add_argument('--k', type=str, default='k5')
    parser.add_argument('--crop', type=str, default='crop4')
    parser.add_argument('--lambda1', type=str, default='1e2')
    parser.add_argument('--lambda2', type=str, default='1e-6', help='number of epochs')
    parser.add_argument('--images_path', type=str, default=None, help='images path to train classification model')
    parser.add_argument('--test_images_path', type=str, default=None, help='images path to test classification model')
    parser.add_argument('--df', type=str, default='/home/unix/mjeon/ECCV2022/privacy/ECCV2022/restyle-same-size-kmeans/ECCV/aptos/nearest_neighbor_centroid/labels/k5_aptos_nearest_df.csv', help='df')
    parser.add_argument('--labels_train', type=str, default='/home/unix/mjeon/k-SALSA/classification_nearest/aptos_val_3000.csv', help='labels for train')
    parser.add_argument('--labels_test', type=str, default='/home/unix/mjeon/data/splited_val/aptos/val_splited.csv', help='labels for test')
    parser.add_argument('--is_centroid', type=str, default='False', help='whether data is centroid or not')
    
    args = parser.parse_args()
    return args


def load_config(path):
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return munchify(cfg)


def copy_config(src, dst):
    shutil.copy(src, dst)


def save_config(config, path):
    with open(path, 'w') as file:
        yaml.safe_dump(config, file)


def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    mean, std = channel_mean.tolist(), channel_std.tolist()
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    return mean, std


def save_weights(model, save_path):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


def print_config(configs):
    for name, config in configs.items():
        print('====={}====='.format(name))
        _print_config(config)
        print('=' * (len(name) + 10))
        print()


def _print_config(config, indentation=''):
    for key, value in config.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indentation, key))
            _print_config(value, indentation + '    ')
        else:
            print('{}{}: {}'.format(indentation, key, value))


def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')


# unnormalize image for visualization
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# convert labels to onehot
def one_hot(labels, num_classes, device, dtype):
    y = torch.eye(num_classes, device=device, dtype=dtype)
    return y[labels]


# convert type of target according to criterion
def select_target_type(y, criterion):
    if criterion in ['cross_entropy', 'kappa_loss']:
        y = y.long()
    elif criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
        y = y.float()
    elif criterion in ['focal_loss']:
        y = y.to(dtype=torch.int64)
    else:
        raise NotImplementedError('Not implemented criterion.')
    return y


# convert output dimension of network according to criterion
def select_out_features(num_classes, criterion):
    out_features = num_classes
    if criterion in regression_loss:
        out_features = 1
    return out_features

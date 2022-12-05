# k-SALSA: k-anonymous synthetic averaging of retinal images via local style alignment
Official implementation (Pytorch) of k-SALSA, (ECCV 2022). ([paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810652.pdf), [supp](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810652-supp.pdf))
> The application of modern machine learning to retinal image analyses offers valuable insights into a broad range of human health conditions beyond ophthalmic diseases. Additionally, data sharing is key to fully realizing the potential of machine learning models by providing a rich and diverse collection of training data. However, the personally-identifying nature of retinal images, encompassing the unique vascular structure of each individual, often prevents this data from being shared openly. While prior works have explored image de-identification strategies based on synthetic averaging of images in other domains (e.g. facial images), existing techniques face difficulty in preserving both privacy and clinical utility in retinal images, as we demonstrate in our work. We therefore introduce $k$-SALSA, a generative adversarial network (GAN)-based framework for synthesizing retinal fundus images that summarize a given private dataset while satisfying the privacy notion of $k$-anonymity. $k$-SALSA brings together state-of-the-art techniques for training and inverting GANs to achieve practical performance on retinal images. Furthermore, $k$-SALSA leverages a new technique, called local style alignment, to generate a synthetic average that maximizes the retention of fine-grain visual patterns in the source images, thus improving the clinical utility of the generated images. On two benchmark datasets of diabetic retinopathy (EyePACS and APTOS), we demonstrate our improvement upon existing methods with respect to image fidelity, classification performance, and mitigation of membership inference attacks. Our work represents a step toward broader sharing of retinal images for scientific collaboration.

## Description   
Official Implementation of our k-SALSA paper for both training and evaluation on downstream tasks. k-SALSA can be applied over different GAN/GAN Inversion frameworks.

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/k-salsa.yaml`.
```
conda env create -f ./environment/k-salsa.yaml
```

## Pretrained Models
In this repository, we provide pretrained ReStyle encoders and StyleGAN2-ADA applied over the 
[pSp](https://github.com/eladrich/pixel2style2pixel) encoders 
across fundus domain.

Please download the pretrained models from the following links.

### ReStyle + pSp
| Path | Description
| :--- | :----------
|[APTOS - ReStyle + pSp]()  | ReStyle applied over pSp trained on the [APTOS](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset.

|[EyePACS - ReStyle + pSp]()  | ReStyle applied over pSp trained on the [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) dataset.

### StyleGAN2-ADA
|[APTOS - StyleGAN2-ADA]()  | StyleGAN2-ADA trained on the [APTOS](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset.

|[EyePACS - StyleGAN2-ADA]()  | StyleGAN2-ADA trained on the [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) dataset.

### Auxiliary Models
In addition, we provide various auxiliary models needed for training your own ReStyle models from scratch.  
This includes the StyleGAN generators and pre-trained models used for loss computation.

| Path | Description
| :--- | :----------
| [ResNet-34 Model](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | ResNet-34 model trained on ImageNet taken from [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) for initializing our encoder backbone.
| [MoCov2 Model](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view) | Pretrained ResNet-50 model trained using MOCOv2 for computing MoCo-based loss on non-facial domains. The model is taken from the [official implementation](https://github.com/facebookresearch/moco).

Note: all StyleGAN models are converted from the official TensorFlow models to PyTorch using the conversion script from [rosinality](https://github.com/rosinality/stylegan2-pytorch).

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. 
However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 

## Training
### Preparing your Data
In order to train k-SALSA on your own data, you should perform the following steps: 
1. Update `configs/paths_config.py` with the necessary data paths and model paths for training and inference.
```
dataset_paths = {
    'train_data': '/path/to/train/data'
    'test_data': '/path/to/test/data',
}
```
2. Configure a new dataset under the `DATASETS` variable defined in `configs/data_configs.py`. There, you should define 
the source/target data paths for the train and test sets as well as the transforms to be used for training and inference.
```
DATASETS = {
	'my_data_encode': {
		'transforms': transforms_config.EncodeTransforms,   # can define a custom transform, if desired
		'train_source_root': dataset_paths['train_data'],
		'train_target_root': dataset_paths['train_data'],
		'test_source_root': dataset_paths['test_data'],
		'test_target_root': dataset_paths['test_data'],
	}
}
```
3. To train with your newly defined dataset, simply use the flag `--dataset_type my_data_encode`.

### Preparing your Generator
In this work, we use rosinality's [StyleGAN2 implementation](https://github.com/rosinality/stylegan2-pytorch). 

If you wish to use your own generator trained using NVIDIA's implementation there are a few options we recommend:
   Using NVIDIA's StyleGAN-ADA PyTorch implementation.  

To train StyleGAN2-ADA, you can follow this [code](https://github.com/dvschultz/stylegan2-ada-pytorch).
At first, you clone the code and create ZIP archive using `dataset_tool.py`

```
python dataset_tool.py --source=./data/aptos_train --dest=./data/aptos_train.zip --width=512 --height=512
```
Then, training StyleGAN2-ADA networks by running this code:
```
python train.py --outdir=./stylegan2ada-aptos --data=./data/aptos_train.zip --gpus=8
```
The pretrained model will be saved in `outdir`

You can then convert the PyTorch `.pkl` checkpoints to the supported format using the conversion script created by [Justin Pinkney](https://github.com/justinpinkney) found in [dvschultz's fork](https://github.com/dvschultz/stylegan2-ada-pytorch/blob/main/SG2_ADA_PT_to_Rosinality.ipynb).  

Once you have the converted `.pt` files, you should be ready to use them in preparing the Encoder.  

### Preparing your Encoder
In this work, we use official code of [ReStyle implementation](https://github.com/yuval-alaluf/restyle-encoder). 
We use the following setting:
- Encoder backbones:
    - We use a ResNet34 encoder backbone using the flags:
        - `--encoder_type=ResNetBackboneEncoder` for pSp
- ID/similarity losses: 
    - Set `--id_lambda=0` and `--moco_lambda=0.5` to use the MoCo-based similarity loss from Tov et al. 
        - Note, you __cannot__ set both `id_lambda` and `moco_lambda` to be active simultaneously.
- You should also adjust the `--output_size` and `--stylegan_weights` flags according to your StyleGAN generator. 

```
python scripts/train_restyle_psp.py \
--dataset_type=aptos_encode \
--encoder_type=ResNetBackboneEncoder \
--exp_dir=experiment/restyle_psp_aptos_encode \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--w_norm_lambda=0 \
--moco_lambda=0.5 \
--input_nc=6 \
--n_iters_per_batch=5 \
--output_size=512 \
--stylegan_weights=pretrained_models/stylegan2-aptos.pt
```

### Run k-SALSA
The main scripts can be found in `k-SALSA_algorithm/scripts/k-SALSA.py`.
The centroid results will be saved to `opts.exp_dir`.
We currently support applying [ReStyle](https://arxiv.org/abs/2104.02699) on the pSp encoder from [Richardson et al. [2020]](https://arxiv.org/abs/2008.00951). 

Applying k-SALSA with the settings used in the paper can be done by running the following commands.

- In the case of k=5:
```
python scripts/k-SALSA.py \
--checkpoint_path=./ckpt/aptos/restyle_100000.pt \
--test_batch_size=5 \
--test_workers=8 \
--n_iters_per_batch=5 \
--data_path=/data/aptos_train \
--exp_dir=./aptos_k5_crop4_1e2_1e-6 \
--num_steps=100 \
--num_crop=4 \
--content_weight=1e2 \
--style_weight=1e-6 \
--same_k=5 \
--df=./data/label/aptos_val_3000.csv \
--data=aptos
```
## Downstream Tasks
### Classification
- [classificaion](https://github.com/JeonMinkyu/k-SALSA/tree/main/classification/pytorch-classification)

### MIA
- [MIA](https://github.com/JeonMinkyu/k-SALSA/tree/main/MIA)

## Credits
**ReStyle model and implementation**
https://github.com/yuval-alaluf/restyle-encoder
Copyright (c) 2021 Yuval Alaluf
License (MIT) https://github.com/yuval-alaluf/restyle-encoder/blob/main/LICENSE

**StyleGAN2 model and implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**IR-SE50 model and implementations:**  
https://github.com/TreB1eN/InsightFace_Pytorch  
Copyright (c) 2018 TreB1eN  
License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

**Ranger optimizer implementation:**  
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer   
License (Apache License 2.0) https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE  

**LPIPS model and implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  

**pSp model and implementation:**   
https://github.com/eladrich/pixel2style2pixel  
Copyright (c) 2020 Elad Richardson, Yuval Alaluf  
License (MIT) https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE

**e4e model and implementation:**   
https://github.com/omertov/encoder4editing
Copyright (c) 2021 omertov  
License (MIT) https://github.com/omertov/encoder4editing/blob/main/LICENSE

**Please Note**: The CUDA files under the [StyleGAN2 ops directory](https://github.com/eladrich/pixel2style2pixel/tree/master/models/stylegan2/op) are made available under the [Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html)

## Acknowledgments
This code borrows heavily from [restyle](https://github.com/yuval-alaluf/restyle-encoder) and 
[stylegan2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch).

## Contact
If you want to contact me:
```
Contact : mjeon@broadinstitute.org
```

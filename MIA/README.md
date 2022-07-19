## Mitigation of Membership Inference Attacks

To compare the privacy properties of the methods, we implemented a well-known class of membership inference attacks (MIA), where an adversary with access to a synthetic dataset attempts to infer whether a target person was part of a specific cluster.

## Training
```
# Start training with: 
CUDA_VISIBLE_DEVICES=0 python MIA_train.py \
--epochs=50 \
--samek=5 \
--centroid_type=centroid_ours \
--data=aptos
```

```
# Test: 
CUDA_VISIBLE_DEVICES=0 python MIA_test.py \
--samek=5 \
--ckpt_epochs=49 \
--centroid_type=centroid_ours \
--ckpt_path=./checkpoint/centroid_ours_aptos_k5_49.pth
```
## Acknowledgments
This code borrows from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

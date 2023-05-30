<!-- Run same size clustering and k-SALSA -->
# Aptos / K=5
CUDA_VISIBLE_DEVICES=0 python scripts/k-SALSA.py \
--checkpoint_path=./ckpt/aptos/restyle_100000_16map2style.pt \
--test_batch_size=5 \
--test_workers=8 \
--n_iters_per_batch=5 \
--data_path=../data/aptos_train \
--exp_dir=./aptos_k5_crop4_1e2_1e-6 \
--num_steps=100 \
--num_crop=4 \
--content_weight=1e2 \
--style_weight=1e-6 \
--same_k=5 \
--df=./data/label/aptos_val_3000.csv \
--data=aptos

<!-- Classification -->
CUDA_VISIBLE_DEVICES=0 python main.py -config ./configs/aptos.yaml --images_path=/home/mjeon/NeurIPS2023/k-salsa/k-SALSA_algorithm/new_exp_aptos_k5_crop4_1e2_1e-6/style_proj --test_images_path=/home/mjeon/data/ECCV_dataset/aptos/val --df=/home/mjeon/NeurIPS2023/k-salsa/k-SALSA_algorithm/save/aptos/labels/k5_nearest_df.csv --labels_train=/home/mjeon/data/ECCV_dataset/aptos/aptos_val_3000.csv --labels_test=/home/mjeon/data/ECCV_dataset/aptos/val_splited.csv --is_centroid=False

<!-- MIA -->
# Train
CUDA_VISIBLE_DEVICES=0 python MIA_train.py \
--epochs=50 \
--samek=5 \
--centroid_type=centroid_ours \
--data=aptos

# Test
train_df_path
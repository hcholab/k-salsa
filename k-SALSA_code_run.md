<!-- Run same size clustering and k-SALSA -->
# Aptos / K=5
CUDA_VISIBLE_DEVICES=0 python scripts/k-SALSA.py \
--checkpoint_path=./checkpoints/iteration_300000.pt \
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
## k-salsa
CUDA_VISIBLE_DEVICES=0 python MIA_train.py --train_df_path=../k-SALSA_algorithm/save/aptos/labels/ --train_image_path=/home/mjeon/data/ECCV_dataset/aptos/aptos_val_3000 --centroid_image_path=../k-SALSA_algorithm/new_exp_aptos_k5_crop4_1e2_1e-6/style_proj/ --centroid_type=centroid_ours --data=aptos

## centroid
CUDA_VISIBLE_DEVICES=2 python MIA_train.py --train_df_path=../k-SALSA_algorithm/save/aptos/labels/ --train_image_path=/home/mjeon/data/ECCV_dataset/aptos/aptos_val_3000 --centroid_image_path=../k-SALSA_algorithm/new_exp_aptos_k5_crop4_1e2_1e-6/centroid/ --centroid_type=centroid --data=aptos


# Test
# k-SALSA
CUDA_VISIBLE_DEVICES=0 python MIA_test.py --centroid_type=centroid_ours --test_df_path=../k-SALSA_algorithm/save/aptos_test/labels/ --ckpt_path=./checkpoint/centroid_ours_aptos_k5_49.pth --test_image_path=/home/mjeon/data/ECCV_dataset/aptos/val/ --centroid_image_path=../k-SALSA_algorithm/TEST_aptos_k5_crop4_1e2_1e-6/style_proj/ --data=aptos
# Centroid
CUDA_VISIBLE_DEVICES=2 python MIA_test.py --centroid_type=centroid --test_df_path=../k-SALSA_algorithm/save/aptos_test/labels/ --ckpt_path=./checkpoint/centroid_aptos_k5_49.pth --test_image_path=/home/mjeon/data/ECCV_dataset/aptos/val/ --centroid_image_path=../k-SALSA_algorithm/TEST_aptos_k5_crop4_1e2_1e-6/centroid/ --data=aptos

# k-SALSA for MIA test
CUDA_VISIBLE_DEVICES=0 python scripts/k-SALSA.py \
--checkpoint_path=./checkpoints/iteration_300000.pt \
--test_batch_size=5 \
--test_workers=8 \
--n_iters_per_batch=5 \
--data_path=/home/mjeon/data/ECCV_dataset/aptos/val \
--exp_dir=./TEST_aptos_k5_crop4_1e2_1e-6 \
--num_steps=100 \
--num_crop=4 \
--content_weight=1e2 \
--style_weight=1e-6 \
--same_k=5 \
--df=/home/mjeon/data/ECCV_dataset/aptos/val_splited.csv \
--data=aptos_test

# FID
python src/pytorch_fid/fid_score.py /home/mjeon/data/ECCV_dataset/aptos/aptos_val_3000 /home/mjeon/NeurIPS2023/k-salsa/k-SALSA_algorithm/new_exp_aptos_k5_crop4_1e2_1e-6/style_proj
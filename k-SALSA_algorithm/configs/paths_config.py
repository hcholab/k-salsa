dataset_paths = {
	'ffhq': '',
	'celeba_test': '',

	'cars_train': '',
	'cars_test': '',

	'church_train': '',
	'church_test': '',

	'horse_train': '',
	'horse_test': '',

	'afhq_wild_train': '',
	'afhq_wild_test': '',
	
	'cifar10_train': '/home/unix/mjeon/privacy/data/CIFAR-10-images/train' ,
	'cifar10_test': '/home/unix/mjeon/privacy/data/CIFAR-10-images/test',

	'aptos2019_train': '/home/unix/mjeon/privacy/data/aptos/train_refined',
	'aptos2019_test': '/home/unix/mjeon/privacy/data/aptos/train_refined',

	'aptos2019_features_train': '/home/unix/mjeon/privacy/data/aptos/train_refined',
	'aptos2019_test': '/home/unix/mjeon/privacy/data/aptos/train_refined',
	
	'odir_train': '/hub_data/privacy/ECCV/data/ODIR/train_val_ODIR/train5000',
	'odir_test': '/hub_data/privacy/ECCV/data/ODIR/train_val_ODIR/val',

	'aptos2019_fundus_train': '/home/unix/mjeon/privacy/data/aptos_fundus/train_val_aptos/train',
	'aptos2019_fundus_test': '/home/unix/mjeon/privacy/data/splited_val/eyepacs/eyepacs_val_10000',

	'eyepacs_train': '/hub_data/privacy/ECCV/data/splited_val/eyepacs/eyepacs_val_10000',
	'eyepacs_test': '/hub_data/privacy/ECCV/data/splited_val/eyepacs/sample1000_val',
}

model_paths = {
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
	'stylegan_church': 'pretrained_models/stylegan2-church-config-f.pt',
	'stylegan_horse': 'pretrained_models/stylegan2-horse-config-f.pt',
	'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
	'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt',
	'dense': '/home/unix/mjeon/denseCL/checkpoint_0199.pth.tar',
	# 'dense': '/home/unix/mjeon/privacy/weights/DenseCL/eyepacs_NoCrop_NoBlur_use_labels_0299.pth.tar',
	# 'moco': '/home/unix/mjeon/privacy/weights/DenseCL/eyepacs_NoCrop_NoBlur_use_labels_0299.pth.tar',
	'stylegan_ada_aptos': 'pretrained_models/refined_5000_aptos.pt',
	'stylegan_ada_aptos_fundus': 'pretrained_models/aptos_fundus_3000.pt',
	'stylegan_ada_eyepacs_fundus': 'pretrained_models/eyepacs_stylegan2_ada_5000.pt'
}

from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	},
	'aptos2019': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['aptos2019_train'],
		'train_target_root': dataset_paths['aptos2019_train'],
		'test_source_root': dataset_paths['aptos2019_test'],
		'test_target_root': dataset_paths['aptos2019_test'],
	},
	'odir_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['odir_train'],
		'train_target_root': dataset_paths['odir_train'],
		'test_source_root': dataset_paths['odir_test'],
		'test_target_root': dataset_paths['odir_test']
	},
	'cifar10_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['cifar10_train'],
		'train_target_root': dataset_paths['cifar10_train'],
		'test_source_root': dataset_paths['cifar10_test'],
		'test_target_root': dataset_paths['cifar10_test'],
	},
	'eyepacs_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['eyepacs_train'],
		'train_target_root': dataset_paths['eyepacs_train'],
		'test_source_root': dataset_paths['eyepacs_test'],
		'test_target_root': dataset_paths['eyepacs_test'],
	},
	'aptos_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['aptos2019_fundus_train'],
		'train_target_root': dataset_paths['aptos2019_fundus_train'],
		'test_source_root': dataset_paths['aptos2019_fundus_test'],
		'test_target_root': dataset_paths['aptos2019_fundus_test'],
	},
	"cars_encode": {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['cars_train'],
		'train_target_root': dataset_paths['cars_train'],
		'test_source_root': dataset_paths['cars_test'],
		'test_target_root': dataset_paths['cars_test']
	},
	"church_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['church_train'],
		'train_target_root': dataset_paths['church_train'],
		'test_source_root': dataset_paths['church_test'],
		'test_target_root': dataset_paths['church_test']
	},
	"horse_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['horse_train'],
		'train_target_root': dataset_paths['horse_train'],
		'test_source_root': dataset_paths['horse_test'],
		'test_target_root': dataset_paths['horse_test']
	},
	"afhq_wild_encode": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['afhq_wild_train'],
		'train_target_root': dataset_paths['afhq_wild_train'],
		'test_source_root': dataset_paths['afhq_wild_test'],
		'test_target_root': dataset_paths['afhq_wild_test']
	},
	"toonify": {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	}
}
from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to ReStyle model checkpoint')
        self.parser.add_argument('--label_path', default=None, type=str,
                                 help='Path to ReStyle model checkpoint')
        self.parser.add_argument('--data_path', type=str, default='gt_images',
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at original output resolution')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=8, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, run on all data')

        # clustering
        self.parser.add_argument('--same_k', type=int, default=5,
                                 help='Number of k for k-anonymity')                
        self.parser.add_argument('--num_steps', type=int, default=100,
                                 help='Number of steps for optimizing k-SALSA')
        self.parser.add_argument('--df', default='/hub_data/privacy/ECCV/data/EyePACs/train_val_eyepacs/labels/new_eyepeac.csv', type=str)
        self.parser.add_argument('--data', default='aptos', type=str)


        # Style
        self.parser.add_argument('--num_crop', type=int, default=4,
                                 help='Number of crops')
        self.parser.add_argument('--style_weight', type=float, default=1e-2,
                                 help='ratio of style loss')
        self.parser.add_argument('--content_weight', type=float, default=1e6,
                                 help='ratio of content loss')

        # arguments for iterative inference
        
        self.parser.add_argument('--n_iters_per_batch', default=5, type=int,
                                 help='Number of forward passes per batch during training.')

        # arguments for encoder bootstrapping
        self.parser.add_argument('--model_1_checkpoint_path', default=None, type=str,
                                 help='Path to encoder used to initialize encoder bootstrapping inference.')
        self.parser.add_argument('--model_2_checkpoint_path', default=None, type=str,
                                 help='Path to encoder used to iteratively translate images following '
                                      'model 1\'s initialization.')

        # arguments for editing
        self.parser.add_argument('--edit_directions', type=str, default='age,smile,pose',
                                 help='comma-separated list of which edit directions top perform.')
        self.parser.add_argument('--factor_ranges', type=str, default='5,5,5',
                                 help='comma-separated list of max ranges for each corresponding edit.')

        # Layerwise
        self.parser.add_argument('--layer_idx', default=1, type=int,
                                 help='Number of forward passes per batch during training.')
    def parse(self):
        opts = self.parser.parse_args()
        return opts

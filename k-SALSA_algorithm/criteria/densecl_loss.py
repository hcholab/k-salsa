import torch
from torch import nn
import torch.nn.functional as F
from configs.paths_config import model_paths
import torchvision.datasets as datasets
import torchvision.models as models

import denseCL.moco.loader
import denseCL.moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class DenseLoss(nn.Module):

    def __init__(self):
        super(DenseLoss, self).__init__()
        print("Loading Dense model from path: {}".format(model_paths["dense"]))

        self.model = self.__load_model()
        self.model.cuda()
        self.model.eval()
        self.criterion = nn.CrossEntropyLoss().cuda()

    @staticmethod
    def __load_model():
        dense = denseCL.moco.builder.MoCo(128, 65536, 0.99, 0.2, True)
        # freeze all layers
        for name, param in dense.named_parameters():
            param.requires_grad = False
        checkpoint_dense = torch.load(model_paths['dense'], map_location="cpu")
        state_dict = checkpoint_dense['state_dict']
        # rename densecl pretrained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        
        dense.load_state_dict(state_dict)
        # dense.load_state_dict(checkpoint_dense['state_dict'])
        return dense

    # def extract_feats(self, x):
    #     x = F.interpolate(x, size=224)
    #     x_feats = self.model(x)
    #     x_feats = nn.functional.normalize(x_feats, dim=1)
    #     x_feats = x_feats.squeeze()
    #     return x_feats

    def forward(self, y_hat, y):
        output, target, ouptut_dense, target_dense = self.model(im_q=y_hat, im_k=y)
        loss = self.criterion(output,target)
        loss_dense = self.criterion(output_dense, target_dense)
        loss = 0.5*(loss+loss_dense)

    # def forward(self, y_hat, y, x):
    #     n_samples = x.shape[0]
    #     x_feats = self.extract_feats(x)
    #     y_feats = self.extract_feats(y)
    #     y_hat_feats = self.extract_feats(y_hat)
    #     y_feats = y_feats.detach()
    #     loss = 0
    #     sim_improvement = 0
    #     sim_logs = []
    #     count = 0
    #     for i in range(n_samples):
    #         diff_target = y_hat_feats[i].dot(y_feats[i])
    #         diff_input = y_hat_feats[i].dot(x_feats[i])
    #         diff_views = y_feats[i].dot(x_feats[i])
    #         sim_logs.append({'diff_target': float(diff_target),
    #                          'diff_input': float(diff_input),
    #                          'diff_views': float(diff_views)})
    #         loss += 1 - diff_target
    #         sim_diff = float(diff_target) - float(diff_views)
    #         sim_improvement += sim_diff
    #         count += 1

    #     return loss / count, sim_improvement / count, sim_logs

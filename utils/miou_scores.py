import os
import numpy as np
import torch
import time
from scipy import linalg # For numpy FID
from pathlib import Path
from PIL import Image
import models.models as models
from utils.fid_folder.inception import InceptionV3
import matplotlib.pyplot as plt
from utils import utils
from utils.drn_segment import drn_105_d_miou
from utils.upernet_segment import upernet101_miou
from utils.deeplabV2_segment import deeplab_v2_miou

# --------------------------------------------------------------------------#
# This code is an adapted version of https://github.com/mseitzer/pytorch-fid
# --------------------------------------------------------------------------#

class miou_pytorch():
    def __init__(self, opt, dataloader_val):
        self.opt = opt
        self.val_dataloader = dataloader_val
        self.best_miou = 0
        self.path_to_save = os.path.join(self.opt.checkpoints_dir, self.opt.name, "MIOU")
        Path(self.path_to_save).mkdir(parents=True, exist_ok=True)

    def compute_miou(self, netG, netEMA,model = None,current_iter = 'latest'):
        image_saver = utils.results_saver_mid_training(self.opt,str(current_iter))
        netG.eval()
        if not self.opt.no_EMA:
            netEMA.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image, label = models.preprocess_input(self.opt, data_i)
                edges = model.module.compute_edges(image)

                label_class_dict = torch.argmax(label, 1).long()
                converted = model.module.coords
                latent = model.module.latent
                if self.opt.no_EMA:
                    generated,_ = netG(label=label,
                                     label_class_dict=label_class_dict,
                                     coords=converted,
                                     latent=latent,
                                     edges=edges,
                                     )
                else:
                    generated,_ = netEMA(label=label,
                                       label_class_dict=label_class_dict,
                                       coords=converted,
                                       latent=latent,
                                       edges=edges,
                                       )
                image_saver(label, generated, data_i["name"])

            if self.opt.dataset_mode == "ade20k":
                answer = upernet101_miou(self.opt.results_dir, self.opt.name, str(current_iter))
            if self.opt.dataset_mode == "cityscapes":
                answer = drn_105_d_miou(self.opt.results_dir, self.opt.name, str(current_iter))
            if self.opt.dataset_mode == "gtavtocityscapes":
                answer = drn_105_d_miou(self.opt.results_dir, self.opt.name, str(current_iter))
            if self.opt.dataset_mode == "coco":
                answer = deeplab_v2_miou(self.opt.results_dir, self.opt.name, str(current_iter))
        netG.train()
        if not self.opt.no_EMA:
            netEMA.train()
        return answer


    def update(self, model, cur_iter):
        print("--- Iter %s: computing MIOU ---" % (cur_iter))
        cur_miou = self.compute_miou(model.module.netG, model.module.netEMA,model,cur_iter)
        self.update_logs(cur_miou, cur_iter)
        print("--- MIOU at Iter %s: " % cur_iter, "{:.2f}".format(cur_miou))
        if cur_miou < self.best_miou:
            self.best_miou = cur_miou
            is_best = True
        else:
            is_best = False
        return is_best

    def update_logs(self, cur_miou, epoch):
        try :
            np_file = np.load(self.path_to_save + "/miou_log.npy")
            first = list(np_file[0, :])
            sercon = list(np_file[1, :])
            first.append(epoch)
            sercon.append(cur_miou)
            np_file = [first, sercon]
        except:
            np_file = [[epoch], [cur_miou]]

        np.save(self.path_to_save + "/miou_log.npy", np_file)

        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(self.path_to_save + "/plot_miou", dpi=600)
        plt.close()


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

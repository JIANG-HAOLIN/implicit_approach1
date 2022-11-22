import os
import numpy as np
import torch
import time
from scipy import linalg # For numpy FID
from pathlib import Path
from PIL import Image
from utils.ade20k_miou.sem.eval import func1

import matplotlib.pyplot as plt
import ntpath



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
        
    def compute_miou(self, netG):
        
        netG.eval()
        
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                label = data_i['A'].cuda()
                image = data_i['B'].cuda()
                generated = netG(label)
                generated = (generated+1)/2
                generated.clamp(0,1)
                im = np.transpose(generated[0].detach().cpu().numpy(), (1, 2, 0))
                im = im*255
                im = Image.fromarray(im.astype(np.uint8))
                path1 ='/no_backups/s1389/CycleGanResize/pytorch-CycleGAN-and-pix2pix/datasets/ADEChallengeData2016/images/validation'
                path = data_i['A_paths']
                short_path = ntpath.basename(path[0])
                name = os.path.splitext(short_path)[0]
                savepath = os.path.join(path1,name+'.jpg')
                im.save(savepath,format = 'JPEG')
            answer = func1()
        netG.train()
        
        return answer


    def update(self, model, cur_iter):
        print("--- Iter %s: computing MIOU ---" % (cur_iter))
        cur_miou = self.compute_miou(model.netG_A)
        self.update_logs(cur_miou, cur_iter)
        print("--- MIOU at Iter %s: " % cur_iter, "{:.2f}".format(cur_miou))
        if cur_miou > self.best_miou:
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
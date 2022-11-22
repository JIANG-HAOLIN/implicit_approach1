import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from tqdm import tqdm as tqdm
from utils.upernet_segment import upernet101_miou
from utils.deeplabV2_segment import deeplab_v2_miou
import torch, os

#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_,_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#


#--- create models ---#
model = models.Unpaired_model_cycle(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()
opt.name = opt.name+"_multimodal"
image_saver = utils.results_saver(opt)
#--- iterate over validation set ---#
for i, data_i in tqdm(enumerate(dataloader_val)):
    _, label = models.preprocess_input(opt, data_i)
    # TODO : change herer to edit the noise vector sampling
    filename = data_i["name"]
    parent = image_saver.path_image
    print(parent)
    for j in range(1,11,1):
        z = torch.randn(label.size(0), opt.z_dim, dtype=torch.float32)
        z = z.view(z.size(0), opt.z_dim, 1, 1)
        z = z.expand(z.size(0), opt.z_dim, label.size(2), label.size(3))
        generated = model(None, label, mode="generate", losses_computer=None)
        iter_filename = [parent + "/" +  filename[0].split("/")[-1].split(".")[0] + "_" +str(j) + ".png"]
        print(iter_filename)
        #input("enter")
        image_saver(label, generated, iter_filename)

#print(deeplab_v2_miou(opt.results_dir,opt.name,opt.ckpt_iter))


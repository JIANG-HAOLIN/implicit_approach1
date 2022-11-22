import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from tqdm import tqdm as tqdm
from utils.upernet_segment import upernet101_miou
from utils.deeplabV2_segment import deeplab_v2_miou


#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_,_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.Unpaired_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

#--- iterate over validation set ---#
for i, data_i in tqdm(enumerate(dataloader_val)):
    _, label = models.preprocess_input(opt, data_i)
    generated = model(None, label, "generate", None)
    image_saver(label, generated, data_i["name"])

print(deeplab_v2_miou(opt.results_dir,opt.name,opt.ckpt_iter))


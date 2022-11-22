import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_pytorch
from utils.miou_scores import miou_pytorch
import matplotlib.backends

##############


import config

# from models import *
from models.generator import ImplicitGenerator as Generator
from models.noise import mixing_noise
import torch
import models.tensor_transforms as tt
# from models.models import *
import numpy as np


# device="cuda"#@jhl







#--- read options ---#
opt = config.read_arguments(train=True)
print("nb of gpus: ", torch.cuda.device_count())
#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloader,dataloader_supervised, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)
fid_computer = fid_pytorch(opt, dataloader_val)
miou_computer = miou_pytorch(opt,dataloader_val)

#--- create models ---#
# model = models.OASIS_model(opt)
# model = models.put_on_multi_gpus(model, opt)






model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
# model = Generator(size=256, hidden_size=512, style_dim=512, n_mlp=8,
#                       activation=None, channel_multiplier=2,
#                       ).to(device)



# loss_G, losses_G_list = model(image, label,"losses_G", losses_computer,converted=converted,latent=noise)


#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))##????????
# optimizerG = torch.optim.Adam(model.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))##????????


def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item



height=256
width =512
in_channel = 1024
out_channel = 512
batch = 1
scale = 1

# input=torch.randn(batch,in_channel,height,width).cuda(0)
# weight= torch.randn(batch,out_channel,in_channel,height,width).cuda(0)
# output= torch.einsum('bihw,boihw->bohw',input,weight).cuda(0)
# print(input.size(),weight.size(),output.size())








#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader_supervised))
if opt.model_supervision != 0 :
    supervised_iter = loopy_iter(dataloader_supervised)

device="cuda"
batch_size = opt.batch_size#@jhl
# label_class_extractor= torch.arange(1,36,1).view(1,35,1,1).cuda(0)

for epoch in range(start_epoch, opt.num_epochs):
    print('epoch %d' %epoch)
    for i, data_i in enumerate(dataloader_supervised):
        print('batch %d' %i)
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader_supervised) + i
        image, label = models.preprocess_input(opt, data_i)

        # label_class_dict = torch.sum((label*label_class_extractor),dim=1,keepdim=True)
        label_class_dict = torch.argmax(label, 1).long()  # [n, h, w]




        # --- generator update ---#

        noise = mixing_noise(batch_size, 512, 0, device)
        coords = tt.convert_to_coord_format(batch_size, 256, 512, integer_values=False).cuda(0)
        # input_img = torch.randn([batch_size, 3, 256, 512])
        input_img = image
        real_stack = torch.cat([input_img, coords], 1).to(device)
        real_img, converted = real_stack[:, :3], real_stack[:, 3:]


        model.module.netG.zero_grad()
        # model.netG.zero_grad()
        # loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)##????????
        loss_G, losses_G_list = model(image=image,
                                      label= label,
                                      label_class_dict=label_class_dict,
                                      mode= "losses_G",
                                      losses_computer= losses_computer,
                                      converted=converted,
                                      latent=noise)



        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()



        # --- discriminator update ---#
        model.module.netD.zero_grad()##????????
        # model.netD.zero_grad()
        # loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = model(image=image,
                                      label= label,
                                      label_class_dict=label_class_dict,
                                      mode= "losses_D",
                                      losses_computer= losses_computer,
                                      converted=converted,
                                      latent=noise)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()

        #--- stats update ---#
        if not opt.no_EMA:
            # utils.update_EMA(model, cur_iter, dataloader_supervised, opt)
            utils.update_EMA(model, cur_iter, dataloader_supervised, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model,
                                     image,
                                     label,
                                     cur_iter,
                                     label_class_dict=label_class_dict,
                                     converted=converted,
                                     latent=noise)
            timer(epoch, cur_iter)
        #if cur_iter % opt.freq_save_ckpt == 0:
        #    utils.save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
            _ = miou_computer.update(model,cur_iter)
        visualizer_losses(cur_iter, losses_G_list+losses_D_list)

#--- after training ---#
utils.update_EMA(model, cur_iter, dataloader_supervised, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")

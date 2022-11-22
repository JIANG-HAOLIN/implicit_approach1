from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import torch
from torch.nn import init
import models.losses as losses
from models.CannyFilter import CannyFilter

from torch import nn, autograd, optim


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        if opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 1:
            self.netG = generators.ImplicitGenerator(opt=opt,size=256, hidden_size=512, style_dim=512, n_mlp=8,
                                  activation=None, channel_multiplier=2).to("cuda")
        else:
            self.netG = generators.OASIS_Generator(opt)
        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()
        #--- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
        self.coords=None
        self.latent=None

    def forward(self, image, label,label_class_dict, mode, losses_computer,converted=None, latent = None,z=None,edges = None):
        # Branching is applied to be compatible with DataParallel
        self.coords=converted

        self.latent=latent
        # print("input latent code:",self.latent)
        if mode == "losses_G":
            loss_G = 0
            fake,_ = self.netG(
                            label=label,
                            label_class_dict=label_class_dict,
                            coords=converted,
                            latent=latent,
                            return_latents=False,
                            truncation=1,
                            truncation_latent=None,
                            input_is_latent=False,
                            )
            # print("fake:",fake.shape,fake)
            output_D = self.netD(fake)
            loss_G_adv = losses_computer.loss(output_D, label, for_real=True)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                # fake = self.netG(label)
                fake, _ = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image)
            loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
                                                                                output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
                else:
                    fake = self.netEMA(
                    label=label,
                    label_class_dict=label_class_dict,
                    coords=converted,
                    latent=latent,
                    return_latents=False,
                    truncation=1,
                    truncation_latent=None,
                    input_is_latent=False,
                )
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            if type(net).__name__ != "ImplicitGenerator":  ########jhl
                net.apply(init_weights)

    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges

class Unpaired_model(nn.Module):
    def __init__(self, opt):
        super(Unpaired_model, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if opt.netG == 1 :
            self.netG = generators.wavelet_generator(opt)
        elif opt.netG == 2 :
            self.netG = generators.wavelet_generator_multiple_levels(opt)
        elif opt.netG == 3 :
            self.netG = generators.wavelet_generator_multiple_levels_no_tanh(opt)
        elif opt.netG == 4:
            self.netG = generators.IWT_spade_upsample_WT_generator(opt)
        elif opt.netG == 5:
            self.netG = generators.wavelet_generator_multiple_levels_reductive_upsample(opt)
        elif opt.netG == 6:
            self.netG = generators.IWT_spade_upsample_WT_reductive_upsample_generator(opt)
        elif opt.netG == 7:
            self.netG = generators.progGrow_Generator(opt)
        elif opt.netG == 8:
            self.netG = generators.ResidualWaveletGenerator(opt)
        elif opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 10:
            self.netG = generators.ResidualWaveletGenerator_2(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator(opt)
            if opt.netDu == 'wavelet':
                self.netDu = discriminators.WaveletDiscriminator(opt)
            elif opt.netDu == 'wavelet_decoder':
                self.netDu = discriminators.WaveletDiscriminator(opt)
                self.wavelet_decoder = discriminators.Wavelet_decoder(opt)
            elif opt.netDu == 'wavelet_decoder_red':
                self.netDu = discriminators.WaveletDiscriminator(opt)
                self.wavelet_decoder = discriminators.Wavelet_decoder_new(opt)
            elif opt.netDu == 'wavelet_decoder_blue':
                self.netDu = discriminators.WaveletDiscriminator(opt)
                self.wavelet_decoder = discriminators.Wavelet_decoder_new()
                self.wavelet_decoder2 = discriminators.BluePart()
            else :
                self.netDu = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.add_edges :
            self.canny_filter = CannyFilter(use_cuda= (self.opt.gpu_ids != -1) )
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.add_edge_loss:
                self.BDCN_loss = losses.BDCNLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        inv_idx = torch.arange(256 - 1, -1, -1).long().cuda()
        label_gc = torch.index_select(label.clone(), 2, inv_idx)
        image_gc = torch.index_select(image.clone(), 2, inv_idx)
        if self.opt.add_edges :
            edges = self.canny_filter(image,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
            import matplotlib.pyplot as plt
            plt.imshow(edges.cpu()[0, 0, ...])
            plt.show()
        else :
            edges = None

        if mode == "losses_G":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            output_D = self.netD(fake)
            loss_G_adv = self.opt.lambda_segment*losses_computer.loss(output_D, label, for_real=True)
            #loss_G_adv = self.opt.lambda_segment*nn.L1Loss(reduction="mean")(output_D[:,:-1,:,:], label)

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

            pred_fake = self.netDu(fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean()
            loss_G += loss_G_GAN

            if self.opt.add_edge_loss:
                loss_G_edge = self.opt.lambda_edge * self.BDCN_loss(label, fake )
                loss_G += loss_G_edge
            else:
                loss_G_edge = None

            return loss_G, [loss_G_adv, loss_G_vgg, loss_G_GAN, loss_G_edge]

        if mode == "losses_G_supervised":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            fake_features = self.netDu(fake,for_features = True)
            real_features = self.netDu(image,for_features = True)

            loss_G_feat = 0
            for real_feat,fake_feat in zip(real_features,fake_features):
                loss_G_feat += self.featmatch(real_feat,fake_feat)

            loss_G += loss_G_feat

            return loss_G,[loss_G_feat]

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=True)
            loss_D += loss_D_fake

            if self.opt.model_supervision == 2 :
                output_D_real = self.netD(image)
                loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
                loss_D += loss_D_real

                if not self.opt.no_labelmix:
                    mixed_inp, mask = generate_labelmix(label, fake, image)
                    output_D_mixed = self.netD(mixed_inp)
                    loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
                                                                                         output_D_fake,
                                                                                         output_D_real)
                    loss_D += loss_D_lm
                else:
                    loss_D_lm = None
            else:
                loss_D_real = None
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "losses_Du":
            loss_Du = 0
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
            output_Du_fake = self.netDu(fake)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            loss_Du += loss_Du_fake

            output_Du_real = self.netDu(image)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()
            loss_Du += loss_Du_real

            # losses_decoder = 0
            # features = self.netDu(image, for_features=True)
            # decoder_output = self.wavelet_decoder(features[0], features[1], features[2], features[3], features[4], features[5])
            # decoder_loss = nn.L1Loss()
            # losses_decoder += decoder_loss(image, decoder_output).mean()
            # loss_Du += losses_decoder

            return loss_Du, [loss_Du_fake,loss_Du_real]



        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake




        if mode == "segment_real":
            segmentation = self.netD(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label,edges = edges)
            else:
                fake = self.netEMA(label,edges = edges)
            segmentation = self.netD(fake)
            return segmentation

        if mode == "Du_regulaize":
            loss_Du = 0
            image.requires_grad = True
            real_pred = self.netDu(image)
            r1_loss = d_r1_loss(real_pred, image).mean()
            loss_Du += 10 * r1_loss
            return loss_Du, [r1_loss]



    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges


    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            self.netDu.load_state_dict(torch.load(path + "Du.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD, self.netDu]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD,]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

class Unpaired_model_gc(nn.Module):
    def __init__(self, opt):
        super(Unpaired_model_gc, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if opt.netG == 1 :
            self.netG = generators.wavelet_generator(opt)
        elif opt.netG == 2 :
            self.netG = generators.wavelet_generator_multiple_levels(opt)
        elif opt.netG == 3 :
            self.netG = generators.wavelet_generator_multiple_levels_no_tanh(opt)
        elif opt.netG == 4:
            self.netG = generators.IWT_spade_upsample_WT_generator(opt)
        elif opt.netG == 5:
            self.netG = generators.wavelet_generator_multiple_levels_reductive_upsample(opt)
        elif opt.netG == 6:
            self.netG = generators.IWT_spade_upsample_WT_reductive_upsample_generator(opt)
        elif opt.netG == 7:
            self.netG = generators.progGrow_Generator(opt)
        elif opt.netG == 8:
            self.netG = generators.ResidualWaveletGenerator(opt)
        elif opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 10:
            self.netG = generators.ResidualWaveletGenerator_2(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            if opt.netDu == 'wavelet':
                self.netDu = discriminators.WaveletDiscriminator(opt)
                self.netDu_gc = discriminators.WaveletDiscriminator(opt)
            else :
                self.netDu = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
                self.netDu_gc = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.add_edges :
            self.canny_filter = CannyFilter(use_cuda= (self.opt.gpu_ids != -1) )
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.add_edge_loss:
                self.BDCN_loss = losses.BDCNLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        inv_idx = torch.arange(256 - 1, -1, -1).long().cuda()
        label_gc = torch.index_select(label.clone(), 2, inv_idx)
        image_gc = torch.index_select(image.clone(), 2, inv_idx)
        if self.opt.add_edges :
            edges = self.canny_filter(image,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
            import matplotlib.pyplot as plt
            plt.imshow(edges.cpu()[0, 0, ...])
            plt.show()
        else :
            edges = None

        if mode == "losses_G_gc":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            pred_fake = self.netDu(fake)
            fake_gc = self.netG(label_gc, edges=edges)
            pred_fake_gc = self.netDu_gc(fake_gc)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean() + self.criterionGAN(pred_fake_gc, True).mean()

            loss_G_adv = self.get_gc_vf_loss(fake, fake_gc)

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            loss_G += loss_G_GAN

            loss_G_vgg = None
            loss_G_edge = None
            return loss_G, [loss_G_adv, loss_G_vgg, loss_G_GAN, loss_G_edge]

        if mode == "losses_Du_gc":
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
                fake_gc = self.netG(label_gc,edges = edges)
            output_Du_fake = self.netDu(fake)
            output_Du_fake_gc = self.netDu_gc(fake_gc)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            loss_Du_fake_gc = self.criterionGAN(output_Du_fake_gc, False).mean()

            output_Du_real = self.netDu(image)
            output_Du_real_gc = self.netDu_gc(image_gc)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()
            loss_Du_real_gc = self.criterionGAN(output_Du_real_gc, True).mean()

            loss_Du = 0.5*(loss_Du_real_gc + loss_Du_fake_gc) + 0.5*(loss_Du_real + loss_Du_fake)

            return loss_Du, [loss_Du_fake,loss_Du_real]


        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake


        if mode == "segment_real":
            segmentation = self.netD(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label,edges = edges)
            else:
                fake = self.netEMA(label,edges = edges)
            segmentation = self.netD(fake)
            return segmentation

        if mode == "Du_regulaize":
            loss_Du = 0
            image.requires_grad = True
            image_gc.requires_grad = True
            real_pred = self.netDu(image)
            real_pred_gc = self.netDu_gc(image_gc)
            r1_loss = d_r1_loss(real_pred, image).mean()
            r1_loss_gc = d_r1_loss(real_pred_gc, image_gc).mean()
            loss_Du += 10 * (r1_loss + r1_loss_gc)
            return loss_Du, [r1_loss]

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0
        criterionGc = torch.nn.L1Loss()
        size = 256

        inv_idx = torch.arange(size-1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * 1.0 #20
        #loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges


    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netDu.load_state_dict(torch.load(path + "Du.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netDu_gc, self.netDu]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG,]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

class Unpaired_model_cycle(nn.Module):
    def __init__(self, opt):
        super(Unpaired_model_cycle, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if opt.netG == 1 :
            self.netG = generators.wavelet_generator(opt)
        elif opt.netG == 2 :
            self.netG = generators.wavelet_generator_multiple_levels(opt)
        elif opt.netG == 3 :
            self.netG = generators.wavelet_generator_multiple_levels_no_tanh(opt)
        elif opt.netG == 4:
            self.netG = generators.IWT_spade_upsample_WT_generator(opt)
        elif opt.netG == 5:
            self.netG = generators.wavelet_generator_multiple_levels_reductive_upsample(opt)
        elif opt.netG == 6:
            self.netG = generators.IWT_spade_upsample_WT_reductive_upsample_generator(opt)
        elif opt.netG == 7:
            self.netG = generators.progGrow_Generator(opt)
        elif opt.netG == 8:
            self.netG = generators.ResidualWaveletGenerator(opt)
        elif opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 10:
            self.netG = generators.ResidualWaveletGenerator_2(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator_cycle(opt)
            if opt.netDu == 'wavelet':
                self.netDu_image = discriminators.WaveletDiscriminator(opt)
                self.netDu_label = discriminators.TileStyleGAN2Discriminator(opt.semantic_nc, opt=opt)
            else :
                self.netDu_image = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
                self.netDu_label = discriminators.TileStyleGAN2Discriminator(opt.semantic_nc, opt=opt)
            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.add_edges :
            self.canny_filter = CannyFilter(use_cuda= (self.opt.gpu_ids != -1) )
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.add_edge_loss:
                self.BDCN_loss = losses.BDCNLoss(self.opt.gpu_ids)

    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        edges = None

        if mode == "losses_G_cycle":
            loss_G = 0
            # Cycle label->image->label
            fake = self.netG(label,edges = edges)
            pred_fake = self.netDu_image(fake)
            cycle_label = self.netD(fake)

            loss_G_GAN = self.criterionGAN(pred_fake, True).mean()

            loss_G_cycle = self.opt.lambda_segment*losses_computer.loss(cycle_label, label, for_real=True)
            #loss_G_cycle = 10.0 * nn.L1Loss(reduction="mean")(cycle_label, label)

            loss_G += loss_G_cycle
            loss_G += loss_G_GAN
            # Cycle 2 image->label->image
            fake_label = self.netD(image)
            pred_fake_label = self.netDu_label(fake_label)
            cycle_image = self.netG(fake_label, None)

            loss_G_GAN2 = self.criterionGAN(pred_fake_label, True).mean()
            loss_G_cycle2 = 10.0 * nn.L1Loss(reduction="mean")(cycle_image, image)
            loss_G += loss_G_GAN2
            loss_G += loss_G_cycle2

            loss_G_vgg = None
            loss_G_edge = None
            return loss_G, [loss_G_cycle, loss_G_vgg, loss_G_GAN, loss_G_edge]

        if mode == "losses_Du_cycle":
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
                fake_label = self.netD(image)

            output_Du_fake = self.netDu_image(fake)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            output_Du_real = self.netDu_image(image)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()

            output_Du_fake_label = self.netDu_label(fake_label)
            loss_Du_fake_label = self.criterionGAN(output_Du_fake_label, False).mean()
            output_Du_real_label = self.netDu_label(label)
            loss_Du_real_label = self.criterionGAN(output_Du_real_label, True).mean()

            loss_Du = 0.5*(loss_Du_real + loss_Du_fake) + 0.5*(loss_Du_real_label + loss_Du_fake_label)

            return loss_Du, [loss_Du_fake,loss_Du_real]


        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake


        if mode == "segment_real":
            segmentation = self.netD(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label,edges = edges)
            else:
                fake = self.netEMA(label,edges = edges)
            segmentation = self.netD(fake)
            return segmentation

        if mode == "Du_regulaize":
            loss_Du = 0
            image.requires_grad = True
            label.requires_grad = True
            real_pred = self.netDu_image(image)
            real_pred_label = self.netDu_label(label)
            r1_loss = d_r1_loss(real_pred, image).mean()
            r1_loss_label = d_r1_loss(real_pred_label, label).mean()
            loss_Du += 10 * (r1_loss + r1_loss_label)
            return loss_Du, [r1_loss]

    def gumbelSampler(self, fake, hard=True, eps=1e-10, dim=1):
        #print(fake)
        logits = torch.log(fake + 0.00001)
        if torch.isnan(logits.max()).data:
            print(fake.min(), fake.max())
        if eps != 1e-10:
            print("`eps` parameter is deprecated and has no effect.")

        gumbels = -(torch.empty_like(logits).exponential_() + eps).log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / 1.0  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = (y_hard - y_soft).detach() + y_soft
            return index.type(torch.cuda.FloatTensor), ret
        else:
            # Reparametrization trick.
            ret = y_soft
            return 0, ret

    def compute_edges(self,images):

        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges


    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netDu_image.load_state_dict(torch.load(path + "Du_image.pth"))
            self.netDu_label.load_state_dict(torch.load(path + "Du_label.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD, self.netDu_image, self.netDu_label]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data['image'], input_semantics


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,)).to("cuda")
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map

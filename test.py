from numpy import generic

import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from utils.fid_scores import fid_pytorch
from utils.drn_segment import drn_105_d_miou as drn_105_d_miou
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import os

generate_images = False
compute_miou_generation = False
compute_fid_generation = False
compute_miou_segmentation_network = True

from models.generator import WaveletUpsample,InverseHaarTransform,HaarTransform,WaveletUpsample2
wavelet_upsample = WaveletUpsample()

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
xfm = DWTForward(J=3, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
ifm = DWTInverse(mode='zero', wave='db3')




from utils.utils import tens_to_im
import numpy as np
from torch.autograd import Variable

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)


from collections import namedtuple



Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )



labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]



#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_,_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
#model = models.Unpaired_model(opt)
model = models.Unpaired_model_cycle(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

mae = []
mse = []

if generate_images :
    #--- iterate over validation set ---#
    for i, data_i in tqdm(enumerate(dataloader_val)):
        _, label = models.preprocess_input(opt, data_i)
        generated = model(None, label, "generate", None).cpu().detach()
        #plt.imshow(tens_to_im(generated[0]))
        #downsampled = torch.nn.functional.interpolate(generated,scale_factor = 0.5)
        #plt.figure()
        #downsampled_wavlet = HaarTransform(3,four_channels=False,levels=1)(downsampled)
        #downsampled_wavlet = HaarTransform(in_channels=3,four_channels=False)(downsampled)

        #upsampled_wavelet = WaveletUpsample2()(downsampled_wavlet)
        #plt.imshow(tens_to_im(downsampled_wavlet[0]))
        #upsampled = InverseHaarTransform(3,four_channels=False,levels=2)(upsampled_wavelet)
        # error = tens_to_im(upsampled[0]-generated[0])-0.5
        #error = tens_to_im(torch.nn.functional.interpolate(torch.nn.functional.interpolate(generated[0],scale_factor = 0.5),scale_factor = 2)-generated[0])-0.5
        #mae.append(np.absolute(error).mean())
        #mse.append(np.sqrt(np.multiply(error,error).mean()))
        #plt.figure()
        #plt.imshow(tens_to_im(upsampled[0]-generated[0]))
        #plt.show()
        image_saver(label, generated, data_i["name"])

#print(np.array(mae).mean())
#print(np.array(mse).mean())



'''print(drn_105_d_miou(opt.results_dir,opt.name,'latest'))
print(drn_105_d_miou(opt.results_dir,opt.name,'20000'))
print(drn_105_d_miou(opt.results_dir,opt.name,'40000'))
print(drn_105_d_miou(opt.results_dir,opt.name,'60000'))'''


if compute_miou_generation :
    print(drn_105_d_miou(opt.results_dir,opt.name,opt.ckpt_iter))
else :
    np_file = np.load(os.path.join(opt.checkpoints_dir,opt.name,'MIOU',"miou_log.npy"))
    first = list(np_file[0, :])
    sercon_miou = list(np_file[1, :])
    #first.append(epoch)
    #sercon.append(cur_fid)
    np_file = [first, sercon_miou]
    print('max miou score is :')
    if opt.ckpt_iter == 'latest' :
        print(sercon_miou[-1])
    elif opt.ckpt_iter == 'best' :
        print(np.max(sercon_miou))
    else :
        print(sercon_miou[first.index(float(opt.ckpt_iter))])

if compute_fid_generation :
    fid_computer = fid_pytorch(opt, dataloader_val)
    fid_computer.fid_test(model)
else :
    np_file = np.load(os.path.join(opt.checkpoints_dir,opt.name,'FID',"fid_log.npy"))
    first = list(np_file[0, :])
    sercon = list(np_file[1, :])
    #first.append(epoch)
    #sercon.append(cur_fid)
    np_file = [first, sercon]
    #print('fid score is :')
    if opt.ckpt_iter == 'latest' :
        print(sercon[-1])
    elif opt.ckpt_iter == 'best' :
        print('min fid : ',np.min(sercon))
        index = sercon.index(np.min(sercon))
        #print(len(sercon))
        #print(len(sercon_miou))
        #print(index)
        print('miou : ',sercon_miou[index])

    else :
        print(sercon[first.index(float(opt.ckpt_iter))])





if compute_miou_segmentation_network :
    hist = np.zeros((35, 35))
    for i, data_i in tqdm(enumerate(dataloader_val)):
        image, label = models.preprocess_input(opt, data_i)
        generated_image = model(image, label, "generate", None)
        #plt.imshow(generated_plot)
        #plt.figure()
        generated = model(image, label, "segment_real", None)
        """generated_entropy = generated[0].cpu()-1
        pixel_entropy = torch.zeros(generated_entropy.size()[1:])
        generated_entropy = generated_entropy.permute((1,2,0))
        generated_entropy = Categorical(logits=generated_entropy).entropy().detach()"""
        '''plt.imshow(generated_entropy)'''

        """for j,row in enumerate(pixel_entropy) :
            for k,_ in enumerate(row) :
                pixel_entropy[j,k] = Categorical(probs= generated_entropy[:,j,k]).entropy()
        #generated_plot = torch.argmax(generated, 1).cpu()
        plt.imshow(pixel_entropy)"""
        generated_plot = torch.argmax(generated,1)[0].cpu()-1
        """original_label = torch.argmax(label,1)[0].cpu()
        error_plot = generated_plot != original_label"""
        '''    plt.figure()
        plt.imshow(error_plot.float())
    
        plt.figure()
        plt.imshow(tens_to_im(generated_image[0]))'''

        """plt.figure()
        plt.imshow(tens_to_im(image.cpu()[0]))
        generated = model(None, label, "generate", None)
        image_saver(label, generated, data_i["name"])
        plt.figure()
        plt.imshow(tens_to_im(generated.cpu()[0]))"""

        hist += fast_hist(generated_plot.flatten().numpy(), torch.argmax(label,1).flatten().cpu().numpy(), 35)

        ious = per_class_iu(hist)
        reduced_ius = np.zeros(19)
        for j,iu in enumerate(ious):
            for gt_label in labels :
                if gt_label.id == j and gt_label.trainId != 255 :
                    reduced_ius[gt_label.trainId] = iu

        print('===> mAP {mAP:.3f}'.format(
            mAP=round(np.nanmean(reduced_ius) * 100, 2)),end="\r")



        if (i % 100) == 0 :
            plt.figure()
            plt.imshow(hist, cmap='hot')
            plt.show()

    ious = per_class_iu(hist) * 100
    reduced_ius = np.zeros(19)
    for i, iu in enumerate(ious):
        for gt_label in labels:
            if gt_label.id == i and gt_label.trainId != 255:
                reduced_ius[gt_label.trainId] = iu

    print(' '.join('{:.03f}'.format(i) for i in reduced_ius))
    print(' '.join('{:.03f}'.format(i) for i in ious))
    print(round(np.nanmean(reduced_ius), 2))
    print(round(np.nanmean(ious), 2))








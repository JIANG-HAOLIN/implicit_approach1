# System libs
import glob
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from utils.ade20k_miou.sem.mit_semseg.config import cfg
from utils.ade20k_miou.sem.mit_semseg.dataset import ValDataset
from utils.ade20k_miou.sem.mit_semseg.models import ModelBuilder, SegmentationModule
from utils.ade20k_miou.sem.mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from utils.ade20k_miou.sem.mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from utils.ade20k_miou.sem.mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    # pbar = tqdm(total=len(loader))
    for batch_data in tqdm(loader):
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    return iou.mean()*100
    '''for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))'''


def main(cfg, gpu,val_list):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)


    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        val_list,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    meanIoU = evaluate(segmentation_module, loader_val, cfg, gpu)

    '''print('Evaluation Done!')'''
    return meanIoU


def upernet101_miou(datadir,name,stage):
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    cfg_path = 'utils/ade20k_miou/sem/config/ade20k-resnet101-upernet.yaml'
    gpuss = 0
    cfg.merge_from_file(cfg_path)


    '''logger = setup_logger(distributed_rank=0)  # TODO
    logger.info("Loaded configuration file {}".format(cfg_path))
    logger.info("Running with config:\n{}".format(cfg))'''

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        "pretrained_models/ade20k-resnet101-upernet",'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        "pretrained_models/ade20k-resnet101-upernet", 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    '''if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))'''

    image_list = sorted(glob.glob(os.path.join(datadir, name, stage, 'image', '*.png')))
    label_list = sorted(glob.glob(os.path.join(datadir, name, stage, 'label', '*.png')))
    assert  len(image_list)==len(label_list)
    validation_list = [{'fpath_img': img_path, 'fpath_segm': seg_path} for img_path, seg_path in
                       zip(image_list, label_list)]

    return main(cfg, gpuss,validation_list)




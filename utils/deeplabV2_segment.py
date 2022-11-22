from __future__ import absolute_import, division, print_function

import torch.hub
import glob
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.coco_miou.deeplab_pytorch.libs.datasets import get_dataset
from utils.coco_miou.deeplab_pytorch.libs.utils import  PolynomialLR, scores


def deeplab_v2_miou(datadir,name,stage):

    image_list = sorted(glob.glob(os.path.join(datadir, name, stage, 'image', '*.png')))
    label_list = sorted(glob.glob(os.path.join(datadir, name, stage, 'label', '*.png')))
    assert len(image_list) == len(label_list)
    validation_list = [{'fpath_img': img_path, 'fpath_segm': seg_path} for img_path, seg_path in
                       zip(image_list, label_list)]

    return 100*test(validation_list)

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device

def test(validation_list):
    """
    Evaluation on validation set
    """
    cuda = '0'
    config_path = 'utils/coco_miou/deeplab_pytorch/configs/cocostuff164k.yaml'

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        files_list = validation_list,
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    ''' model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)'''
    model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182)
    model.load_state_dict(torch.load('pretrained_models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth'))
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)
    # Path to save logits
    '''logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    makedirs(logit_dir)
    print("Logit dst:", logit_dir)

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores.json")
    print("Score dst:", save_path)'''

    preds, gts = [], []
    for image_ids, images, gt_labels in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)

        # Save on disk for CRF post-processing
        '''for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())'''

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        preds += list(labels.cpu().numpy())
        gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)
    print('mIoU: ', score["Mean IoU"])
    return score["Mean IoU"]

    '''with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)'''


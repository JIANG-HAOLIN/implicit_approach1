import os

from utils.miou_folder import drn
import  math
from torch import nn
import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image
from os.path import exists, join, split
import glob
from torchvision import datasets, transforms
import json
from PIL import Image


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


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

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



def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


def test(eval_data_loader, model, num_classes, has_gt=True):
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    for iter, (image, label) in enumerate(eval_data_loader):
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        if has_gt:
            label = label.numpy()[0,0,:,:]
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            '''            print('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)),end="\r")'''
        print('Eval: [{0}/{1}]'.format(iter, len(eval_data_loader)),end="\r")
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        print(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)

class MIoUDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,name,stage):
        self.image_list = sorted(glob.glob(os.path.join(data_dir,name,stage,'image','*.png')))
        self.label_list = sorted(glob.glob(os.path.join(data_dir,name,stage,'label','*.png')))

        info = json.load(open(join(data_dir, 'info.json'), 'r'))
        normalize = transforms.Normalize(mean=info['mean'],
                                         std=info['std'])
        self.transforms = transforms.Compose([
                transforms.ToTensor(), normalize])
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        data = Image.open(self.image_list[index])
        data = self.transforms(data)

        if self.label_list is not None:
            input_label = Image.open(self.label_list[index])
            input_label = np.array(input_label)[:, :, 0]

            label_copy =  np.empty(input_label.shape, dtype=np.uint8)
            for label_tuple in labels:
            	label_copy[input_label == label_tuple.id] = label_tuple.trainId
            	
            input_label = Image.fromarray(label_copy)
            input_label = self.to_tensor(label_copy)*255
            resize = False
            if resize :
                data = data.unsqueeze(0)
                data = torch.nn.functional.interpolate(data,scale_factor= 0.5, mode='bilinear')
                data = data.squeeze(0)

                input_label = input_label.unsqueeze(0)
                input_label = torch.nn.functional.interpolate(input_label,scale_factor= 0.5, mode='nearest')
                input_label = input_label.squeeze(0)




        return data,input_label

    def __len__(self):
        return len(self.image_list)


def drn_105_d_miou(datadir,name,stage):
    model = DRNSeg('drn_d_105',19,pretrained=False)
    model.load_state_dict(torch.load('pretrained_models/drn-d-105_ms_cityscapes.pth'))
    model = torch.nn.DataParallel(model).cuda()

    dataset = MIoUDataset(datadir,name,stage)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False
    )
    return  test(test_loader,model,19)


def drn_22_d_miou(datadir,name,stage):
    model = DRNSeg('drn_d_22',19,pretrained=False)
    model.load_state_dict(torch.load('pretrained_models/drn_d_22_cityscapes.pth'))
    model = torch.nn.DataParallel(model).cuda()

    dataset = MIoUDataset(datadir,name,stage)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False
    )
    return  test(test_loader,model,19)



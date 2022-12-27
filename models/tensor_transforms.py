import random

import torch

device_tensor_transforms = 'cuda'


def convert_to_coord_format(b, h, w, device=device_tensor_transforms, integer_values=False):
    if integer_values:  ##源代码有误？？ 应是arrange w repeat h？？
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)

    else:## x 横轴 y 纵轴 long：x_channel = torch.linspace(-2, 2, w, device=device).view(1, 1, 1, -1).repeat(b, 1, h, 1)
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, h, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, w)
    return torch.cat((x_channel, y_channel), dim=1)


def random_crop(tensor, size):
    assert tensor.dim() == 4, tensor.shape  # frames x channels x h x w
    h, w = tensor.shape[-2:]
    h_start = random.randint(0, h - size) if h - size > 0 else 0
    w_start = random.randint(0, w - size) if w - size > 0 else 0
    return tensor[:, :, h_start: h_start + size, w_start: w_start + size]


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        return random_crop(tensor, self.size)


def random_horizontal_flip(tensor):
    flip = random.randint(0, 1)
    if flip:
        return tensor.flip(-1)
    else:
        return tensor


def identity(tensor):
    return tensor

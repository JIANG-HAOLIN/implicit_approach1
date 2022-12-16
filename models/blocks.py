import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

import gc


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            # self.weight = self.weight.to('cuda')
            # print(input.device, self.weight.device, )
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        approach= -1,
        add_dist = False
    ):
        super().__init__()
        self.approach = approach
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.add_dist = add_dist

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        if self.approach == 0:
            self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
            self.clade_weight_modulation = EqualLinear(style_dim, self.out_channel, bias_init=1) #jhl
            self.clade_bias_modulation = EqualLinear(style_dim, self.out_channel, bias_init=1) #jhl
        elif self.approach == 1:
            self.param_free_norm = nn.InstanceNorm2d(self.in_channel, affine=False)  ##jhl
            self.clade_weight_modulation = EqualLinear(style_dim*2, self.out_channel, bias_init=1)  # jhl
            self.clade_bias_modulation = EqualLinear(style_dim*2, self.out_channel, bias_init=1)  # jhl
        elif self.approach == 2:
            pass
        else:
            pass

        if self.add_dist :
            if (self.approach == 0 or self.approach == 1) :
                self.dist_conv_w = nn.Conv2d(2, 1, kernel_size=1, padding=0)
                nn.init.zeros_(self.dist_conv_w.weight)
                nn.init.zeros_(self.dist_conv_w.bias)
                self.dist_conv_b = nn.Conv2d(2, 1, kernel_size=1, padding=0)
                nn.init.zeros_(self.dist_conv_b.weight)
                nn.init.zeros_(self.dist_conv_b.bias)
            elif self.approach ==2 :
                pass
            else:
                pass




    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style,label_class_dict=None,label = None,class_style = None,dist_map = None):
        batch, in_channel, height, width = input.shape

        # style = self.modulation(style).view(batch, 1, in_channel, 1, 1)##[1,1,1024,1,1,]
        # ###modulation(style):35x512=>35x1024
        # weight = self.scale * self.weight * style
        # ##[1,512,1024,1,1]
        # if self.demodulate:
        #     demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        #     weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        #     ##[1,512,1024,1,1]
        # weight = weight.view(
        #     batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        # )##[1,512,1024,1,1]

        if self.upsample:
            print('upsample')
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
            ###modulation(style):35x512=>35x1024
            weight = self.scale * self.weight * style
            ##[1,512,1024,1,1]
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
                ##[1,512,1024,1,1]
            weight = weight.view(
                batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )  ##[1,512,1024,1,1]


            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)


        elif self.downsample:
            print('downsample')
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
            ###modulation(style):35x512=>35x1024
            weight = self.scale * self.weight * style
            ##[1,512,1024,1,1]
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
                ##[1,512,1024,1,1]
            weight = weight.view(
                batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )  ##[1,512,1024,1,1]


            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:

            # ###pixel interation
            # input = input.reshape(height, width, batch,1, in_channel)
            # out   = []
            # for h in range(height):
            #     for w in range(width):
            #         print(h)
            #         pixel_input = input[h][w].reshape(1, batch * in_channel, 1, 1)
            #         LUT=int(label_class_dict[h][w]-1)
            #         class_style = style[LUT]
            #         class_style = self.modulation(class_style).view(batch, 1, in_channel, 1, 1)
            #         weight = self.scale * self.weight * class_style
            #         if self.demodulate:
            #             demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            #             weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
            #             ##[1,512,1024,1,1]
            #         weight = weight.view(
            #             batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)  ##[1,512,1024,1,1]
            #         pixel_input = pixel_input.reshape(1, batch * in_channel, 1, 1)
            #
            #         out.append(F.conv2d(pixel_input, weight, padding=self.padding, groups=batch))
            # out = torch.Tensor(out)
            #         # print(out)
            # _,height, width, _ , _ = out.shape
            # out = out.view(batch, self.out_channel, height, width)


            # ###class iteration
            # out = torch.zeros(batch, self.out_channel, height, width).cuda(0)
            # # style = style.reshape(35,512)
            # for class_index in range(35):
            #     print(class_index)
            #     class_map = label_class_dict
            #     class_map[class_map != class_index] = -1.000
            #     class_map[class_map == class_index] = 1.000
            #     class_map[class_map == -1] = 0.000
            #     class_map = class_map.view(batch, 1, 256, 512).cuda(0)
            #     input = (input * class_map).cuda(0)
            #     class_style = style[class_index].view(batch, 1, 512).cuda(0)
            #     class_style = self.modulation(class_style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
            #     ###modulation(style):35x512=>35x1024
            #     weight1 = self.scale * self.weight * class_style.cuda(0)
            #     ##[1,512,1024,1,1]
            #     if self.demodulate:
            #         demod = torch.rsqrt(weight1.pow(2).sum([2, 3, 4]) + 1e-8)
            #     weight1 = weight1 * demod.view(batch, self.out_channel, 1, 1, 1).cuda(0)
            #     ##[1,512,1024,1,1]
            #     weight1 = weight1.view(
            #         batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)  ##[1,512,1024,1,1]
            #     print(input.device,input.double().device,weight1.device,weight1.double().device,out.device)
            #     input = input.view(1, batch * in_channel, height, width)
            #     out = out + F.conv2d(input.double().cuda(0), weight1.double().cuda(0), padding=self.padding, groups=batch)
            #     del weight1

            #     gc.collect()
            #     print("xxxxxxxxxxxxxxxxx")
            # _, _, height, width = out.shape
            # out = out.view(batch, self.out_channel, height, width)


            ## Modulation+CLADE layer
            if self.approach == 0:
                style = self.modulation(style).view(batch, 1, in_channel, 1, 1)  ##[1,1,1024,1,1,]
                ###modulation(style):35x512=>35x1024
                weight = self.scale * self.weight * style
                ##[1,512,1024,1,1]
                if self.demodulate:
                    demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                    weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
                ##[1,512,1024,1,1]
                weight = weight.view(
                    batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)  ##[1,512,1024,1,1]

                input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight, padding=self.padding, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)
                ##apply CLADE layer
                clade_weight_init = self.clade_weight_modulation(class_style)
                clade_bias_init = self.clade_bias_modulation(class_style)
                out = self.param_free_norm(out)
                class_weight = F.embedding(label_class_dict, clade_weight_init).permute(0, 3, 1, 2)
                #before permute:[n, h, w, c] after permute [n, c, h, w]
                class_bias = F.embedding(label_class_dict, clade_bias_init).permute(0, 3, 1, 2)
                # before permute:[n, h, w, c] after permute [n, c, h, w]

                if self.add_dist:
                    # input_dist = F.interpolate(dist_map, size=input.size()[2:], mode='nearest')
                    # class_weight = class_weight * (1 + self.dist_conv_w(input_dist))
                    # class_bias = class_bias * (1 + self.dist_conv_b(input_dist))

                    input_dist = dist_map
                    class_weight= class_weight.to('cpu')
                    class_bias= class_bias.to('cpu')
                    alpha_weight = (1 + self.dist_conv_w(input_dist)).to('cpu')
                    print(alpha_weight.shape)
                    alpha_bias = 1 + self.dist_conv_b(input_dist).to('cpu')
                    class_weight = (class_weight * alpha_weight).to('cuda')
                    class_bias = (class_bias * alpha_bias).to('cuda')

                out = out * class_weight + class_bias


            ##class_label_dict = [N, H, W]

            ## use only CLADE layer, no mod or demod
            if self.approach == 1:
                # print('approach 1 activated')
                weight = self.weight.view(self.out_channel,in_channel,self.kernel_size,self.kernel_size) ##[512,1024,1,1]

                # input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight, padding=self.padding)
                # _, _, height, width = out.shape
                # out = out.view(batch, self.out_channel, height, width)

                out = self.param_free_norm(out)

                style = style.view(batch, 1, 512).expand(batch, 35, 512)
                class_style = class_style.view(1, 35, 512).expand(batch, 35, 512)
                style_concatenation = torch.cat((style, class_style), dim=2).view(batch*35, 1024)
                clade_weight_init = self.clade_weight_modulation(style_concatenation).view(batch, 35, self.out_channel)
                clade_bias_init = self.clade_bias_modulation(style_concatenation).view(batch, 35, self.out_channel)
                class_weight = torch.einsum('nic,nihw->nchw', clade_weight_init, label)
                class_bias = torch.einsum('nic,nihw->nchw', clade_bias_init, label)

                if self.add_dist:
                    # input_dist = F.interpolate(dist_map, size=input.size()[2:], mode='nearest')
                    #
                    # class_bias = class_bias * (1 + self.dist_conv_b(input_dist))

                    input_dist = dist_map
                    class_weight = class_weight.to('cpu')
                    class_bias = class_bias.to('cpu')
                    alpha_weight = (1 + self.dist_conv_w(input_dist)).to('cpu')
                    print(alpha_weight.shape)
                    alpha_bias = 1 + self.dist_conv_b(input_dist).to('cpu')
                    class_weight = (class_weight * alpha_weight).to('cuda')
                    class_bias = (class_bias * alpha_bias).to('cuda')

                out = out * class_weight + class_bias


            ## brutal Matrix Computation approach
            if self.approach == 2:
                # print('apply approach 2 : Matrix Computation')
                style = style.view(batch, 1, 512)
                class_style = class_style.view(1, 35, 512)
                style_addition = style + class_style ##[N, 35, 512]
                style_addition = style_addition.view(batch*35, 512)
                style_weight_init = self.modulation(style_addition).view(batch, 35, self.in_channel)
                pixel_class_style = torch.einsum('nci,nchw->nihw',style_weight_init,label)
                weight = self.weight.view(self.out_channel, self.in_channel)
                weight_per_pixel = self.scale * (torch.einsum('oi,nihw->noihw',weight , pixel_class_style))
                if self.demodulate:
                    demod = torch.rsqrt(torch.sum(weight_per_pixel.pow(2), dim=2, keepdim=True) + 1e-8)
                    weight_per_pixel = weight_per_pixel * demod
                out = torch.einsum('nihw,noihw->nohw', input, weight_per_pixel)


        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def  __init__(self, channel, size=None,):
        super().__init__()
        # print(size)
        self.learnable_vectors = nn.Parameter(torch.randn(1, channel, size[0], size[1]))
        # print(self.learnable_vectors.shape)

    def forward(self, input):
        batch = input.shape[0]
        out = self.learnable_vectors.repeat(batch, 1, 1, 1)
        # print(out.shape)
        ##output = [batch,channel,size.512]
        return out



class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        activation=None,
        downsample=False,
        approach=-1,
        add_dist = False
    ):
        super().__init__()
        self.add_dist = add_dist
        self.approach = approach
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            downsample=downsample,
            approach=self.approach,
            add_dist= self.add_dist
        )

        self.activation = activation
        self.noise = NoiseInjection()
        if activation == 'sinrelu':
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
            self.activate = ScaledLeakyReLUSin()
        elif activation == 'sin':
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
            self.activate = SinActivation()
        else:
            self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None,label_class_dict=None,label=None,class_style=None,dist_map = None):
        out = self.conv(input, style,label_class_dict=label_class_dict,label=label,class_style=class_style,dist_map=dist_map)
        out = self.noise(out, noise=noise)
        if self.activation == 'sinrelu' or self.activation == 'sin':
            out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1],approach=-1,add_dist = False):#jhl
        super().__init__()
        self.add_dist = add_dist
        self.approach = approach
        self.upsample = upsample
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False,
                                    approach=self.approach,add_dist=self.add_dist)#jhl
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None,label_class_dict=None,label=None,class_style=None,dist_map=None):
        out = self.conv(input, style,
                        label_class_dict=label_class_dict,label=label,class_style=class_style,dist_map=dist_map)
        out = out + self.bias

        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out


class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        upsample=False,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], kernel_size=3, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
        self.conv2 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        # print('initialsation:',self.conv.weight.device)

        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        # print('forwarding',self.conv.weight.device,x.device)
        return self.conv(x)


class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class LFF(nn.Module):
    def __init__(self, hidden_size, ):
        super(LFF, self).__init__()
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x


class ScaledLeakyReLUSin(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out_lr = F.leaky_relu(input[:, ::2], negative_slope=self.negative_slope)
        out_sin = torch.sin(input[:, 1::2])
        out = torch.cat([out_lr, out_sin], 1)
        return out * math.sqrt(2)


class StyledResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, blur_kernel=[1, 3, 3, 1], demodulate=True,
                 activation=None, upsample=False, downsample=False):
        super().__init__()

        self.conv1 = StyledConv(in_channel, out_channel, kernel_size, style_dim,
                                demodulate=demodulate, activation=activation)
        self.conv2 = StyledConv(out_channel, out_channel, kernel_size, style_dim,
                                demodulate=demodulate, activation=activation,
                                upsample=upsample, downsample=downsample)

        if downsample or in_channel != out_channel or upsample:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False, upsample=upsample,
            )
        else:
            self.skip = None

    def forward(self, input, latent):
        out = self.conv1(input, latent)
        out = self.conv2(out, latent)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out = (out + skip) / math.sqrt(2)

        return out

## @jhl new

# class upscale_Interpolate(nn.Module):
#     def __init__(self, size, mode):
#         super(upscale_Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode
#
#     def forward(self, x):
#         x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
#         return x

def make_dist_train_val_cityscapes_datasets(mask_batch=None,dir = '/home/tzt/dataset/cityscapes/',norm='norm'):
    # label_dir = os.path.join(dir, 'gtFine')
    # phases = ['val','train']
    # for phase in phases:
    #     if 'test' in phase:
    #         continue
    #     print('process',phase,'dataset')
    #     citys = sorted(os.listdir(os.path.join(label_dir,phase)))
    #     for city in citys:
    #         label_path = os.path.join(label_dir, phase, city)
    #         label_names_all = sorted(os.listdir(label_path))
    #         label_names = [p for p in label_names_all if p.endswith('_labelIds.png')]
    #         for label_name in label_names:
    #             print(label_name)
    #             mask = np.array(Image.open(os.path.join(label_path, label_name)))
                # check_mask(mask)
    batch_size = mask_batch.shape[0]
    dist_cat_np_batch = []
    for i in range(batch_size):
        mask = np.array(mask_batch[i,:,:])##(256,512)
        h_offset, w_offset = cal_connectedComponents(mask, norm)
        dist_cat_np = np.concatenate((h_offset[np.newaxis, ...], w_offset[np.newaxis, ...]), 0)
    # dist_name = label_name[:-12]+'distance.npy'
    # np.save(os.path.join(label_path, dist_name), dist_cat_np)
        dist_cat_np_batch.append(dist_cat_np)
    return torch.Tensor(np.array(dist_cat_np_batch))


def cal_connectedComponents(mask, normal_mode='norm'):
    label_idxs = np.unique(mask)
    H, W = mask.shape
    out_h_offset = np.float32(np.zeros_like(mask))
    out_w_offset = np.float32(np.zeros_like(mask))
    for label_idx in label_idxs:
        if label_idx == 0:
            continue
        tmp_mask = np.float32(mask.copy())
        tmp_mask[tmp_mask!=label_idx] = -1
        tmp_mask[tmp_mask==label_idx] = 255
        tmp_mask[tmp_mask==-1] = 0
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(tmp_mask))
        connected_numbers = len(centroids)-1
        for c_idx in range(1,connected_numbers+1):
            tmp_labels = np.float32(labels.copy())
            tmp_labels[tmp_labels!=c_idx] = 0
            tmp_labels[tmp_labels==c_idx] = 1
            h_offset = (np.repeat(np.array(range(H))[...,np.newaxis],W,1) - centroids[c_idx][1])*tmp_labels
            w_offset = (np.repeat(np.array(range(W))[np.newaxis,...],H,0) - centroids[c_idx][0])*tmp_labels
            h_offset = normalize_dist(h_offset, normal_mode)
            w_offset = normalize_dist(w_offset, normal_mode)
            out_h_offset += h_offset
            out_w_offset += w_offset

    return out_h_offset, out_w_offset

def normalize_dist(offset, normal_mode):
    if normal_mode == 'no':
        return offset
    else:
        return offset / np.max(np.abs(offset)+1e-5)

def show_results(ins):
    plt.imshow(ins)
    plt.show()

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def check_mask(mask, check_idx = 255):
    idx = np.unique(mask)
    if check_idx in idx:
        print(idx)
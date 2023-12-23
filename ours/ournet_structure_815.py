"""
"""
# import sys
#
# sys.path.append("/home/aistudio/code")
import numpy as np
import math
from function import gaussian_kernel
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.nn.utils import spectral_norm
import time
from function import extract_image_patches, \
    reduce_mean, reduce_sum, same_padding, lap_conv

up_ratio = 2  # 基本不变
kernelsize_temp = 3
kernelsize_temp2 = 5  # 空间注意力细节程度，越大细节越大
padding_mode = 'circular'
pi = 3.14159265

# from methods.Pfnet.Pfnet import log
# nn.initializer.set_global_initializer(nn.initializer.Normal(mean=0.0, std=1))
# nn.initializer.set_global_initializer(nn.initializer.Uniform())
# nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.LeakyReLU()):

        m = [default_conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class simple_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernelsize: int = 3):
        super(simple_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, [kernelsize, kernelsize], stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU())

    def forward(self, x: torch.Tensor):
        return self.net(x)


class basic_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(basic_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, [kernelsize, kernelsize], stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, output_channel, [kernelsize, kernelsize], stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(output_channel))

    def forward(self, x: torch.Tensor):
        return self.conv2(self.conv1(x))


class res_net_nobn(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(res_net_nobn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, [kernelsize, kernelsize], stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Conv2d(mid_channel, output_channel, [kernelsize, kernelsize], stride=1,
                               padding_mode=padding_mode,
                               padding=int(kernelsize // 2))

    def forward(self, x: torch.Tensor):
        temp = self.conv1(x)
        temp2 = self.conv2(temp)
        return temp2  # + x


class res_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(res_net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, [kernelsize, kernelsize], stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Conv2d(mid_channel, output_channel, [kernelsize, kernelsize], stride=1,
                               padding_mode=padding_mode,
                               padding=int(kernelsize // 2))

    def forward(self, x: torch.Tensor):
        temp = self.conv1(x)
        temp2 = self.conv2(temp)
        return temp2 + x


# 得到高光谱的high level特征
class encoder_hs(nn.Module):
    def __init__(self, band_in, ks=5, ratio=4, len_res=5, mid_channel=64):
        super(encoder_hs, self).__init__()
        self.ratio = ratio

        self.conv = nn.Sequential(  # 处理全色
            nn.Conv2d(band_in, mid_channel, [ks, ks], padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channel, mid_channel, [ks, ks], padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(mid_channel))

        self.res0 = nn.ModuleList([res_net(mid_channel, mid_channel, mid_channel=mid_channel,
                                          kernelsize=ks) for _ in range(len_res)])

    def forward(self, hs):
        x2 = self.conv(hs)

        for i in range(len(self.res0)):
            x2 = self.res0[i](x2)

        return x2


# 得到多光谱的high level特征
class encoder_ms(nn.Module):
    def __init__(self, band_in, ks=5, ratio=4, len_res=5, mid_channel=64):
        super(encoder_ms, self).__init__()
        self.ratio = ratio

        self.conv = nn.Sequential(  # 处理全色
            nn.Conv2d(band_in, int(mid_channel / 2), [ks, ks], padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(int(mid_channel / 2)),
            nn.LeakyReLU(),
            nn.Conv2d(int(mid_channel / 2), mid_channel, [ks, ks], padding=int(ks / 2), padding_mode='circular'),
            nn.BatchNorm2d(mid_channel))

        # self.dense = Dense_block(ks=3, mid_channel=mid_channel, len_dense=5)

        # self.att = attention(ks=3, mid_ch=mid_channel)

        # self.dense2 = Dense_block(ks=3, mid_channel=mid_channel, len_dense=5)

        self.res0 = nn.ModuleList([res_net(mid_channel, mid_channel, mid_channel=mid_channel,
                                          kernelsize=ks) for _ in range(len_res)])
        # self.act = nn.Tanh()

    def forward(self, ms):
        x2 = self.conv(ms)
        # x2 = self.dense2(self.att(self.dense(x0)))

        for i in range(len(self.res0)):
            x2 = self.res0[i](x2)

        return x2


# 三个上采样融合网络具有相同的设计但是不同的参数
class Dense_block(nn.Module):
    def __init__(self, ks=3, mid_channel=64, len_dense=5):
        super(Dense_block, self).__init__()

        self.resnet = nn.ModuleList([res_net_nobn(mid_channel * (i + 1), mid_channel,
                                                 mid_channel=mid_channel, kernelsize=ks) for i in
                                    range(len_dense)])  # 修改

        self.down_layer = simple_net(mid_channel * (len_dense + 1), mid_channel)

    def forward(self, x):
        temp_result = self.resnet[0](x)
        result = torch.cat((x, temp_result), 1)

        for i in range(1, 5):
            temp_result = self.resnet[i](result)
            result = torch.cat((result, temp_result), 1)

        return self.down_layer(result) + x


class attention(nn.Module):
    def __init__(self, ks=3, mid_ch=64):
        super(attention, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, [ks, ks], padding=int(ks / 2), padding_mode='circular'),
            nn.LeakyReLU(),
            nn.Conv2d(mid_ch, mid_ch, [ks, ks], padding=int(ks / 2), padding_mode='circular'))

        self.spe_att = nn.Sequential(
            nn.Conv2d(mid_ch, int(mid_ch / 2), [1, 1]),
            nn.LeakyReLU(),
            nn.Conv2d(int(mid_ch / 2), mid_ch, [1, 1]),
            nn.Sigmoid()
        )
        self.spa_att = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, [1, 1]),
            nn.Sigmoid())

    def forward(self, x):
        x0 = self.conv0(x)
        return self.spe_att(x0) * x0 + self.spa_att(x0) * x0 + x


class cross_scale_attention(nn.Module):
    def __init__(self, in_ch, band_hs, output_pad=0, ks=5, mid_ch=64, ratio=4, stride=4, softmax_scale=10):
        super(cross_scale_attention, self).__init__()

        self.scale = ratio
        self.stride = stride
        self.ks = ks
        self.softmax_scale = softmax_scale
        self.mid_ch = mid_ch
        self.band_hs = band_hs
        self.in_ch = in_ch
        self.output_pad = output_pad # if (ratio % 2) == 1 else 0
        # self.shifts = shifts

        self.conv_q = BasicBlock(self.in_ch, self.mid_ch, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())
        self.conv_k = BasicBlock(self.in_ch, self.mid_ch, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())
        self.conv_v = BasicBlock(band_hs, band_hs, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())
        self.conv_result = BasicBlock(band_hs, band_hs, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())

    def forward(self, ms, pan, pan2):  # ms为原始分辨率多光谱影像或多光谱细节
        # 处理k
        k_fea = self.conv_k(pan2)
        N, _, h, w = k_fea.shape

        k_patch = extract_image_patches(k_fea, ksizes=[self.ks, self.ks],
                                        strides=[self.stride, self.stride], rates=[1, 1], padding='same')

        k_patch = k_patch.view(N, self.mid_ch, self.ks, self.ks, -1).permute(0, 4, 1, 2, 3)
        k_group = torch.split(k_patch, 1, dim=0)

        # 处理q
        q_fea = self.conv_q(pan)
        q_group = torch.split(q_fea, 1, dim=0)  # 作为被卷积的对象

        # 处理v
        v_fea = self.conv_v(ms)
        v_patch = extract_image_patches(v_fea, ksizes=[self.ks, self.ks],
                                        strides=[self.stride, self.stride], rates=[1, 1], padding='same')

        v_patch = v_patch.view(N, self.band_hs, self.ks, self.ks, -1).permute(0, 4, 1, 2, 3)
        v_group = torch.split(v_patch, 1, dim=0)

        result = []
        softmax_scale = self.softmax_scale

        for q, k, v in zip(q_group, k_group, v_group):
            k0 = k[0]
            k0_max = torch.max(torch.sqrt(reduce_sum(torch.pow(k0, 2), axis=[1, 2, 3],
                                                       keepdim=True)))
            k0 = k0 / k0_max

            # print(k0.shape)
            q0 = q[0]
            # print(q0.shape)
            q0 = same_padding(torch.unsqueeze(q0, 0), ksizes=[self.ks, self.ks], strides=[self.stride, self.stride],
                              rates=[1, 1])
            weight = F.conv2d(q0, k0, stride=self.stride)

            weight_norm = F.softmax(weight * softmax_scale, dim=1)

            v0 = v[0]
            # print(weight_norm.shape)
            deconv_map = F.conv_transpose2d(weight_norm, v0, stride=self.stride, output_padding=self.output_pad,
                                            padding=1)

            deconv_map = deconv_map / 6

            result.append(deconv_map)

        result = torch.cat(result, dim=0)
        return self.conv_result(result)

class Conv_spe(nn.Module):
    def __init__(self, band_hs, band_ms):
        super(Conv_spe, self).__init__()
        """
        convolution operation on spectral/band dimension. The output attention map is compute on global spatial field.
        input:  1*C*H*W
        filter: 1*c*H*W
        output: C*c
        """
        self.band_ms = band_ms
        self.band_hs = band_hs

    def forward(self, hs, ms):

        # assert ms.shape[2] == hs.shape[2]
        result = []

        for i in range(0, self.band_ms):
            result.append(F.conv2d(hs, torch.tile(ms[i:(i+1), :, :, :], [self.band_hs, 1, 1, 1]), stride=ms.shape[2],
                                   groups=self.band_hs))
        # print(result[0].shape)
        return torch.cat(result, 0)

class cross_scale_attention_spe(nn.Module):
    def __init__(self, band_hs, band_ms, ks=5, mid_ch=64, ratio=4, stride=4, softmax_scale=10):
        super(cross_scale_attention_spe, self).__init__()

        self.ratio = ratio
        self.stride = stride
        self.ks = ks
        self.softmax_scale = softmax_scale
        self.mid_ch = mid_ch
        self.band_hs = band_hs
        self.band_ms = band_ms
        # self.in_ch = in_ch
        self.output_pad = 1 if (ratio % 2) == 0 else 0

        self.spe_conv = Conv_spe(band_hs, band_ms)
        self.conv_q = BasicBlock(self.band_ms, self.band_ms, kernel_size=3, stride=ratio**2, bias=True, bn=False,
                                 act=nn.LeakyReLU())
        self.conv_k = nn.Sequential(  # 处理高光谱
            # nn.Upsample(scale_factor=ratio, mode='bicubic'),
            BasicBlock(self.band_hs, self.band_hs, kernel_size=3, stride=ratio, bias=True, bn=False, act=nn.LeakyReLU())
        )  # upsample the HSI

        self.conv_v = BasicBlock(band_ms, band_ms, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())

        self.conv_result = BasicBlock(band_hs, band_hs, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU())

    def forward(self, hrms, msi, hsi):  # hrms=pan, msi=pan2, hsi= ms2
        N, _, _, _ = hsi.shape
        # hh = int(h / self.ratio)

        hrms_group = torch.split(self.conv_v(hrms), 1, dim=0)
        ms_f_group = torch.split(self.conv_q(msi), 1, dim=0)

        hs_f = self.conv_k(hsi)
        # hs_f_expand = torch.cat([hs_f, hs_f[:, :self.band_ms - 1, :, :]], dim=1)
        # hs_unfold = [hs_f_expand[:, i:(i + self.band_ms), :, :] for i in range(self.band_hs)]
        # hs_unfold = torch.cat(hs_unfold, dim=3)
        # print(hs_unfold.shape)
        hs_f_group = torch.split(hs_f, 1, dim=0)

        result = []
        for hrms, ms_f, hs_f in zip(hrms_group, ms_f_group, hs_f_group):
            # hs_f0 = hs_f[0]
            # hs_f0 = torch.cat([hs_f0, hs_f0[:, :self.band_ms, :, :]], dim=1)
            # hs_unfold = [hs_f0[:, i:(i+self.band_ms), :, :] for i in range(self.band_hs)]       # 移到循环外
            # hs_unfold = torch.stack(hs_unfold, dim=3)
            # print(hs_unfold.shape)

            k0_max = torch.max(torch.sqrt(reduce_sum(torch.pow(ms_f, 2), axis=[2, 3],
                                                       keepdim=True)))
            ms_f = (ms_f / k0_max).permute(1, 0, 2, 3)
            # hs_f0 = hs_f[0]
            # print(hs_f.shape)
            # print(ms_f.shape)

            att_map = self.spe_conv(hs_f, ms_f)
            # print(att_map.shape)
            # print(att_map.shape)    # 1 * ms_band * 1 * hs_band
            # att_map = att_map.permute(1, 3, 0, 2])
            att_map = F.softmax(att_map * self.softmax_scale, dim=1)

            # hrms0 = hrms[0]
            deconv_map = F.conv_transpose2d(hrms, att_map)  # ks = 1
            # print(hrms.shape)
            # print(att_map.shape)
            # print(deconv_map.shape)
            deconv_map = deconv_map / 6

            result.append(deconv_map)

        result = torch.cat(result, dim=0)
        return self.conv_result(result)


class recon(nn.Module):
    def __init__(self, band_hs, ks=3, mid_ch=64):
        super(recon, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(band_hs, band_hs, [1, 1]),
            nn.Conv2d(band_hs, band_hs, [ks, ks], padding=int(ks / 2), padding_mode='circular'),
            nn.Tanh())

    def forward(self, x):
        return self.conv0(x)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class Gray(nn.Module):
    def __init__(self, in_channel=4, retio=4):
        super(Gray, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # weight_attr0 = torch.ParamAttr(name="weight",
        #                                 trainable=True)
        # weight_attr1 = torch.ParamAttr(name="weight1",
        #                                 trainable=True)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel * retio),
            nn.ReLU(),
            nn.Linear(in_channel * retio, in_channel),
            nn.Sigmoid(),
            nn.Softmax()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        y = self.avg(x2).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out = torch.sum(out, dim=1, keepdim=True)
        # out = torch.cat([out for _ in range(self.in_channel)], dim=1)

        return out

class Our_net(nn.Module):
    def __init__(self, band_ms, mid_ch=10, ratio=16):  # band is the number of hyperspectral image
        super(Our_net, self).__init__()
        # self.band_hs = band_hs
        self.band_ms = band_ms
        self.ratio = ratio

        sig = 1 # (1 / (2 * 2.7725887 / 16 ** 2)) ** 0.5  # 1.0
        ks = 5
        kernel3 = torch.tensor(np.multiply(gaussian_kernel(ks, sig),
                            gaussian_kernel(ks, sig).T), dtype=torch.float32)
        kernel_ms0 = torch.tile(torch.reshape(kernel3, [1, 1, ks, ks]), [band_ms, 1, 1, 1])
        # kernel_ms2 = np.tile(np.reshape(kernel3, [1, 1, 3, 3]), [1, band_ms, 1, 1])
        kernel_pan = torch.reshape(kernel3, [1, 1, ks, ks])

        self.blur_ms14 = nn.Conv2d(band_ms, band_ms, [ks, ks], stride=1, padding=int(ks/2), padding_mode='circular', groups=band_ms)
        self.blur_ms14.weight.data = kernel_ms0

        self.down_ms14 = nn.AvgPool2d([1, 1], stride=ratio)

        self.blur_pan14 = nn.Conv2d(1, 1, [ks, ks], stride=1, padding=int(ks/2), padding_mode='circular')
        self.blur_pan14.weight.data = kernel_pan

        self.down_pan14 = nn.AvgPool2d([1, 1], stride=ratio)

        # model = Our_net(band_ms, mid_ch=mid_ch, ratio=ratio)
        self.encoder_ms_net = encoder_hs(band_ms, ks=3, ratio=ratio, len_res=5, mid_channel=mid_ch)
        self.encoder_pan_net = encoder_hs(1, ks=3, ratio=ratio, len_res=5, mid_channel=mid_ch)
        # self.encoder_ms2_net = encoder_hs(band_ms, ks=3, ratio=ratio, len_res=5, mid_channel=mid_ch)
        # self.encoder_hs_net_red2 = encoder_hs(band_hs, ks=3, ratio=ratio, len_res=5, mid_channel=mid_ch)
        # self.encoder_ms_net = encoder_ms(band_ms, ks=3, ratio=ratio, len_res=5, mid_channel=mid_ch)

        # decoder_hs_net = decoder_hs(band_hs, ks=5, mid_channel=mid_ch)
        # decoder_ms_net = decoder_ms(band_ms, ks=5, mid_channel=mid_ch)
        unfold_size = 3
        self.transformer = cross_scale_attention(ks=unfold_size, in_ch=mid_ch, band_hs=band_ms, mid_ch=mid_ch, ratio=ratio,
                                                 stride=unfold_size, softmax_scale=10)
        # self.transformer_red = cross_scale_attention(ks=unfold_size, in_ch=mid_ch, band_hs=band_hs, mid_ch=mid_ch, ratio=ratio,
        #                                              stride=3, softmax_scale=10)
        # self.transformer_red2 = cross_scale_attention(ks=unfold_size, in_ch=mid_ch, band_hs=band_hs, mid_ch=mid_ch, ratio=ratio,
        #                                               stride=3, softmax_scale=10)

        self.transformer_spe = cross_scale_attention_spe(band_ms, 1, ks=5, mid_ch=mid_ch, ratio=ratio, softmax_scale=10)

        self.down_dim = nn.Sequential(
            nn.Conv2d(band_ms+band_ms, band_ms+band_ms, kernel_size=11, padding=5, padding_mode=padding_mode),
            nn.Conv2d(band_ms+band_ms, band_ms, kernel_size=1))

    def forward(self, ms, pan):

        blur_ms = self.blur_ms14(ms)
        ms2 = self.down_ms14(blur_ms)  # 真值
        blur_pan = self.blur_pan14(pan)
        pan2 = self.down_pan14(blur_pan)  # 真值

        ms_f = self.encoder_ms_net(ms)
        pan_f = self.encoder_pan_net(pan)
        ms_fuse2 = self.transformer(ms, pan_f, ms_f)
        ms_fuse22 = self.transformer_spe(pan, pan2, ms2)

        # np.save('/home/aistudio/result/result_mid/' + 'result_spa' + str(0) + '.npy',
        #                         ms_fuse2[0, :, :, :])
        # np.save('/home/aistudio/result/result_mid/' + 'result_spe' + str(0) + '.npy',
        #                         ms_fuse22[0, :, :, :])

        # blur_ms_fuse2 = self.blur_ms14(ms_fuse2)
        # ms_fuse2_416 = self.down_ms14(blur_ms_fuse2)
        # pan_srf_down2 = torch.mean(ms_fuse2, 1, keepdim=True)

        ms_srf_down = torch.mean(ms, 1, keepdim=True)  # 和pan2算loss

        # 原始分辨率
        ms_fuse0 = self.down_dim(torch.cat([ms_fuse2, ms_fuse22], 1)) \
                        + F.upsample(ms, scale_factor=self.ratio, mode='bicubic')
        blur_ms_fuse0 = self.blur_ms14(ms_fuse0)
        ms_fuse0_14 = self.down_ms14(blur_ms_fuse0)  # 用于算loss

        pan_srf_down0 = torch.mean(ms_fuse0, 1, keepdim=True) 

        return ms_fuse0, ms2, pan2, ms_srf_down, ms_fuse0_14, pan_srf_down0, blur_ms, blur_pan, blur_ms_fuse0


class dis(nn.Module):
    def __init__(self, band_ms):
        super(dis, self).__init__()
        padding_mode = 'circular'
        norm = spectral_norm
        self.conv0 = norm(nn.Conv2d(band_ms, 16, kernel_size=3, padding=1, padding_mode=padding_mode))

        self.conv = nn.ModuleList([norm(nn.Conv2d(2 ** (i + 4), 2 ** (i + 5), kernel_size=3,
                                                 padding=1, padding_mode=padding_mode)) for i in range(4)])

        self.conv1 = nn.Sequential(
            norm(nn.Conv2d(256, 1, kernel_size=3, padding=1, padding_mode=padding_mode)),
            nn.Sigmoid())

    def forward(self, ms):
        f0 = self.conv0(ms)
        for i in range(4):
            f0 = self.conv[i](f0)
        return self.conv1(f0)


if __name__ == '__main__':

    a = torch.randn([10, 8, 48, 48])
    b = torch.randn([10, 3, 192, 192])
    # c = torch.randn([10, 8, 144, 144])

    transformer = cross_scale_attention_spe(8, 3, ratio=4)

    d = transformer(b, b, a)
    print(d.shape)

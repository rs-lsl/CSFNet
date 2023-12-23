import torch.nn as nn
from torch import Tensor

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class L1_Charbonnier_loss(nn.Module):
    """
    L1 Charbonnierloss
    """

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss

# 实现Sam loss
class SAMLoss(torch.nn.Module):
    
   def __init__(self):
       super(SAMLoss, self).__init__()
   def forward(self, input, label):
       return _sam(input, label)

def _sam(img1, img2):

    inner_product = torch.sum(img1 * img2, dim=0)
    img1_spectral_norm = torch.sqrt(torch.sum(img1 ** 2, dim=0))
    img2_spectral_norm = torch.sqrt(torch.sum(img2 ** 2, dim=0))
    # numerical stability
    cos_theta = torch.clip(inner_product / (img1_spectral_norm * img2_spectral_norm + 1e-10), min=0, max=1)
    return torch.mean(torch.acos(cos_theta))


# TV loss //Total variance 待改进
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.shape[0]
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.shape[1]*t.shape[2]*t.shape[3]

def cc_loss(x, y):
        """输入两个向量计算相关系数"""

        x_reducemean = x - torch.mean(x, dim=0)
        y_reducemean = y - torch.mean(y, dim=0)

        fenzi = torch.mean(x_reducemean * y_reducemean)

        return 1 - fenzi / (torch.std(x_reducemean) * torch.std(y_reducemean))

class laplacian_sharp(nn.Module):
    def __init__(self, channels=5):
        super(laplacian_sharp, self).__init__()
        self.channels = channels
        # kernel = [[1, -1],
        #           [-1, 1]]

        kernel = [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]

        kernel_size = 3
        kernel = np.reshape(kernel, [1,1,kernel_size,kernel_size])
        kernel = np.repeat(kernel, self.channels, dim=0)
        weight_attr0 = torch.ParamAttr(initializer=nn.initializer.Assign(kernel), trainable=False)
        self.conv = nn.Conv2d(self.channels, self.channels,
                              kernel_size=kernel_size, weight_attr=weight_attr0,
                              groups=channels, padding=int(kernel_size/2), padding_mode='circular')

    def forward(self, x):
        return self.conv(x)
        

class Uiqi(nn.Module):
    def __init__(self, block_size=32, band_ms=5):
        super(Uiqi, self).__init__()
        self.block_size = block_size

        kernel = np.ones((block_size, block_size)) / (block_size ** 2)
        kernel = np.reshape(kernel, [1, 1, block_size, block_size])
        kernel = np.tile(kernel, [band_ms, 1, 1, 1])
        weight_attr0 = torch.ParamAttr(initializer=nn.initializer.Assign(kernel), trainable=False)
        self.avg_conv = nn.Conv2d(band_ms, band_ms, block_size, padding=int(block_size/2),
                                  padding_mode='circular', weight_attr=weight_attr0, groups=band_ms)

    def forward(self, img1, img2):  # b * 1 * w * h
        mu1 = self.avg_conv(img1)
        mu2 = self.avg_conv(img2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.avg_conv(img1) - mu1_sq
        sigma2_sq = self.avg_conv(img2) - mu2_sq
        sigma12 = self.avg_conv(img1 * img2) - mu1_mu2

        qindex_map = np.ones(sigma12.shape)
        sigma1_sq = sigma1_sq.numpy()
        sigma2_sq = sigma2_sq.numpy()
        sigma12 = sigma12.numpy()
        mu1_sq = mu1_sq.numpy()
        mu2_sq = mu2_sq.numpy()
        mu1_mu2 = mu1_mu2.numpy()

        idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
        qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
        # sigma !=0 and mu == 0
        idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
        qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
        # sigma != 0 and mu != 0
        idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
        qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
                (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

        return np.mean(qindex_map)

class Uiqi_pan(nn.Module):
    def __init__(self, block_size=32):
        super(Uiqi_pan, self).__init__()
        self.block_size = block_size

        kernel = np.ones((block_size, block_size)) / (block_size ** 2)
        kernel = np.reshape(kernel, [1, 1, block_size, block_size])
        weight_attr0 = torch.ParamAttr(initializer=nn.initializer.Assign(kernel), trainable=False)
        self.avg_conv = nn.Conv2d(1, 1, block_size, padding=int(block_size/2),
                                  padding_mode='circular', weight_attr=weight_attr0)

    def forward(self, img1, pan):  # b * 1 * w * h
        pan = (pan - torch.mean(pan)) / torch.std(pan)
        pan = pan * torch.std(pan) + torch.mean(img1)

        mu1 = self.avg_conv(img1)
        mu2 = self.avg_conv(pan)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.avg_conv(img1) - mu1_sq
        sigma2_sq = self.avg_conv(pan) - mu2_sq
        sigma12 = self.avg_conv(img1 * pan) - mu1_mu2

        qindex_map = np.ones(sigma12.shape)
        sigma1_sq = sigma1_sq.numpy()
        sigma2_sq = sigma2_sq.numpy()
        sigma12 = sigma12.numpy()
        mu1_sq = mu1_sq.numpy()
        mu2_sq = mu2_sq.numpy()
        mu1_mu2 = mu1_mu2.numpy()

        idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
        qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
        # sigma !=0 and mu == 0
        idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
        qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
        # sigma != 0 and mu != 0
        idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
        qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
                (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

        return np.mean(qindex_map)

def Q(a,  b): # N x H x W
    E_a = torch.mean(a, dim=(1,2))
    E_a2 = torch.mean(a * a, dim=(1,2))
    E_b = torch.mean(b, dim=(1,2))
    E_b2 = torch.mean(b * b, dim=(1,2))
    E_ab = torch.mean(a * b, dim=(1,2))

    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b

    return torch.mean(4 * cov_ab * E_a * E_b / (var_a + var_b) / (E_a**2 + E_b**2))

def D_lambda(ps, l_ms): # N x C x H x W
    L = ps.shape[1]
    sum = 0

    for i in range(L):
        for j in range(L):
            if j!=i:
                sum += torch.abs(Q(ps[:,i,:,:], ps[:,j,:,:]) - Q(l_ms[:,i,:,:], l_ms[:,j,:,:]))
    
    return sum/L/(L-1)

def D_s(ps, l_ms, pan, l_pan): # N x C x H x W
    L = ps.shape[1]

    sum = 0

    for i in range(L):
        sum += torch.abs(Q(ps[:,i,:,:], pan[:,0,:,:]) - Q(l_ms[:,i,:,:], l_pan[:,0,:,:]))

    return sum/L

def qnr(c_D_lambda, c_D_s, alpha=1, beta=1):
    """QNR - No reference IQA"""  # The higher the QNR index, the better the quality of the fused product
    QNR_idx = (1 - c_D_lambda) ** alpha * (1 - c_D_s) ** beta
    return QNR_idx

def no_ref_evaluate(fused_image0, test_ms_image, test_pan_image, data_range=255):
    # no reference metrics

    D_lambda0 = D_lambda(fused_image0 / data_range, Tensor(test_ms_image))
    l_pan = F.interpolate(Tensor(test_pan_image), size=[test_ms_image.shape[2], test_ms_image.shape[3]], mode='bilinear')
    D_s0 = D_s(fused_image0 / data_range, Tensor(test_ms_image), Tensor(test_pan_image), l_pan)
    c_qnr = qnr(D_lambda0, D_s0)

    return [D_lambda0.numpy(), D_s0.numpy(), c_qnr]

class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()      # nn.BCELoss
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":
            self.loss = self._wgan_loss
        elif self.gan_type == "wgan_softplus":
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ["wgan", "wgan_softplus"]:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return torch.ones(input.shape) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == "hinge":
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        return loss

# from torch.vision.models import vgg19
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """
    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = vgg19(pretrained=True, batch_norm=False)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt)

        return torch.linalg.norm(x_features - gt_features, p='fro') 

        # calculate perceptual loss
        # if self.perceptual_weight > 0:
        #     percep_loss = 0
        #     for k in x_features.keys():
        #         if self.criterion_type == 'fro':
        #             percep_loss += torch.linalg.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
        #         else:
        #             percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        #     percep_loss *= self.perceptual_weight
        # else:
        #     percep_loss = None

        # return percep_loss

    def _gram_mat(self, x):
        n, c, h, w = x.shape
        features = x.view(n, c, w * h)
        features_t = features.permute(0, 2, 1)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


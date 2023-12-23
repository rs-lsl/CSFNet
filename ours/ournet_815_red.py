# -*- coding: utf-8 -*-
"""
License: MIT
@author: lsl
E-mail: 987348977@qq.com
去掉了大部分bicubic相关的loss
"""
import sys

sys.path.append("/home/aistudio/code")
sys.path.append("/home/aistudio/code/pansharpening/ours")
sys.path.append("/home/aistudio/code/pansharpening/Pan-GAN")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import time
import numpy as np

# from Pfnet_structure_real import Pf_net  # 还有 Pfnet_structure2
from ours.ournet_structure_815 import Our_net  # 还有 Pfnet_structure2
from function import lap_conv
from ours.ournet_dataset import Mydata2#, Mydata_test
from loss_function import Uiqi, Uiqi_pan, cc_loss#, Weight_l1_loss# , D_lambda, D_s
from loss_function import SSIM
from metrics import psnr
from function import initialize_weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#  hyper parameters
test_batch_size = 1


def ournet(train_ms_image, train_pan_image, train_label,
           test_ms_image, test_pan_image, test_label, num_epochs=50,
           mid_ch=20, batch_size=10, learning_rate = 1e-3,
           ratio=4, name='wv2'):
    # 影像维数
    # hs_data是真反射率数据
    num_epochs = 9 if name == 'gf2' else 250
    band_ms = test_ms_image.shape[1]
    num_patch = test_ms_image.shape[0]

    print(train_ms_image.shape)
    print(test_ms_image.shape)

    # torch.seed(1500)
    #  定义数据和模型
    dataset0 = Mydata2(train_ms_image, train_pan_image, train_label)
    train_loader = torch.utils.data.DataLoader(dataset0, num_workers=0, batch_size=batch_size,
                                        shuffle=True, drop_last=True)

    dataset1 = Mydata2(test_ms_image, test_pan_image, test_label)
    test_loader = torch.utils.data.DataLoader(dataset1, num_workers=0, batch_size=test_batch_size,
                                       shuffle=False, drop_last=False)

    model = Our_net(band_ms, mid_ch=mid_ch, ratio=ratio)
    initialize_weights(model)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    # print('# parameters:', sum(param.numel() for param in model.parameters()))

    L2_loss = nn.L1Loss()
    ssim_ms = SSIM(window_size=11, size_average=True, channel=band_ms)
    ssim_pan = SSIM(window_size=11, size_average=True,
                    channel=1)  # ssim_loss = SSIMLoss(window_size=11, sigma=1.5, data_range=1, channel=band)

    pre_train = False
    if pre_train:
        model_dict =torch.load('/home/aistudio/result/parameters/ournet_model_'+name+'.pdparams')    # 加载训练好的模型参数
        model.set_state_dict(model_dict)

    loss_save = []
    loss_save_test = []

    element_num_ms = batch_size * np.prod(train_ms_image.shape[1:]) * ratio ** 2
    element_num_pan = batch_size * np.prod(train_pan_image.shape[2:]) * band_ms

    # scheduler = optim.lr_scheduler_scheduler.MultiStepLR(learning_rate, [30, 60], gamma=0.5, verbose=False)
    # scheduler = optim.lr_scheduler.StepDecay(learning_rate, 50, gamma=0.1, verbose=False)
    # scheduler = optim.lr_scheduler.CosineAnnealingDecay(2e-3, 10, verbose=False)
    optimizer = optim.Adam(lr=learning_rate, params=model.parameters())

    for epoch in range(num_epochs):
        time0 = time.time()
        loss_total = 0.0
        loss_total_test = 0.0
        # psnr0 = 0.0

        model.train()

        for i, (ms, pan, _) in enumerate(train_loader):
            optimizer.zero_grad()
            ms = ms.to(device, dtype=torch.float32)
            pan = pan.to(device, dtype=torch.float32)

            ms_fuse0, ms2, pan2, ms_srf_down, ms_fuse0_14, pan_srf_down0, blur_ms, blur_pan, blur_ms_fuse0 = model(ms, pan)

            # cc loss
            ms_up = F.upsample(ms, scale_factor=ratio, mode='nearest')
            ms_diff = (ms_fuse0 - ms_up).reshape(element_num_ms)

            pan2_up = F.upsample(pan2, scale_factor=ratio, mode='nearest')
            pan_diff = (torch.tile((pan - pan2_up), [1, band_ms, 1, 1])).reshape(element_num_pan)

            loss_cc = cc_loss(ms_diff, pan_diff)

            ms_up = F.upsample(ms, scale_factor=ratio, mode='bicubic')
            loss_bicubic_high = L2_loss(ms_up, ms_fuse0)

            ssim_total_spa = 1 - ssim_pan(pan_srf_down0, pan)
            ssim_total_spe = 1 - ssim_ms(ms_fuse0_14, ms)


            loss = loss_cc*10 + \
            loss_bicubic_high*10 + \
            ssim_total_spe*50 + ssim_total_spa*10

            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        # scheduler.step()

        if ((epoch + 1) % 10) == 0:
            psnr0 = 0.0

            model.eval()
            with torch.no_grad():
                for (images_ms, images_pan, image_label) in test_loader:
                    outputs_temp = model(images_ms, images_pan)
                    psnr0 += psnr(outputs_temp[0].cpu().numpy(), image_label.cpu().numpy())
            print('epoch %d of %d, using time: %.2f , loss of train: %.4f, Psnr: %.4f' %
                  (epoch + 1, num_epochs, time.time() - time0, loss_total, psnr0 / num_patch))

            # loss_save_test.append(loss_total_test)
        loss_save.append(loss_total)

    torch.save(model.state_dict(), '/home/aistudio/result/parameters/ournet_model_red_'+name+'.pth')

    #测试模型
    model_dict =torch.load('/home/aistudio/result/parameters/ournet_model_red_'+name+'.pth')    # 加载训练好的模型参数
    # print(model_dict.keys())
    model.set_state_dict(model_dict)

    model.eval()
    image_all = []
    # label_all = []
    with torch.no_grad():

        time_all = []
        for (images_hs, images_ms, _) in test_loader:
            time0 = time.time()
            outputs_temp = model(images_hs, images_ms)
            time_all.append(time.time() - time0)
            # outputs = outputs_temp.cpu()
            image_all.append(outputs_temp[0])
        print("ournet:", np.mean(np.asarray(time_all)))

        a = torch.cat(image_all, dim=0)
        # print((base[0].parameters())[0])

    return a

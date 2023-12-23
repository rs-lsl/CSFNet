
import numpy as np
import pandas as pd
import os
import time
# import cv2
import torch
import torch.nn.functional as F

from metrics import ref_evaluate
from metrics import no_ref_evaluate

from ours.ournet_815_full import ournet

if __name__ == '__main__':

    train_wv2 = False
    train_gf2 = True
    if train_wv2:
        # band_ms = [1, 2, 3, 4, 5]   # wv2
        from save_image_wv2 import generate_data, crop_data
        name = 'wv2'
        train_num = 2000
        test_num = 200
        num_epochs = 50  # 10比2的虽然定量差，但视觉效果好

        print('Testing worldview2 dataset!')
    elif train_gf2:
        # band_ms = [0, 1, 2, 3]
        from save_image_gf2 import generate_data, crop_data
        name = 'gf2'
        train_num = 1000
        test_num = 100
        num_epochs = 50

        print('Testing gaofen2 dataset!')
    else:
        print(1)

    mid_ch = 32     # endmember nums  30
    batch_size = 1
    learning_rate = 1e-4

    ms, pan = generate_data()
    ms_crop, pan_crop = crop_data(ms, pan)
    # ms_crop = np.random.random([1000,4,16,16])
    # pan_crop = np.random.random([1000,1,64,64])
    
    index = np.arange(ms_crop.shape[0])
    np.random.seed(1000)
    np.random.shuffle(index)
    # 随机shuffle
    train_ms_image = ms_crop[index[:train_num], :, :, :]
    train_pan_image = pan_crop[index[:train_num], :, :, :]

    test_ms_image = ms_crop[index[-test_num:], :, :, :]
    test_pan_image = pan_crop[index[-test_num:], :, :, :]

    print(train_ms_image.shape)
    print(train_pan_image.shape)

    print(test_ms_image.shape)
    print(test_pan_image.shape)

    print(np.max(test_ms_image))
    print(np.max(test_pan_image))

    ratio = int(test_pan_image.shape[2] / test_ms_image.shape[2])
    print('ratio: ', ratio)

    '''setting save parameters'''
    save_images = False
    save_num = 10  # 存储测试影像数目
    # save_channels = [1, 2, 4]  # BGR for hyperspectral image
    save_dir = []
    for i7 in range(save_num):
        save_dir.append('/home/aistudio/result/result_pansharpening_full_'+name+'/results' + str(i7) + '/')
        if save_images and (not os.path.isdir(save_dir[i7])):
            os.makedirs(save_dir[i7])

    '''定义度量指标和度量函数'''
    no_ref_results = {}
    no_ref_results.update({'metrics: ': '  D_lamda, D_s,    QNR'})
    len_ref_metrics = 7
    len_no_ref_metrics = 3
    result = []
    result_diff = []
    metrics_result_no_ref = []  # 存储测试影像指标

    test_ournet = True

    '''Pfnet method'''
    if test_ournet:
        print('evaluating ournet method')
        fused_image = ournet(
            train_ms_image, train_pan_image,
            test_ms_image, test_pan_image,
            num_epochs=num_epochs, mid_ch=mid_ch, learning_rate=learning_rate,
            batch_size=batch_size, ratio=ratio, name=name)

        fused_image = fused_image.numpy()
        # ref_results_all = []
        if save_images:
            for j5 in range(save_num):
                np.save(save_dir[j5] + 'our_result_'+ name + str(j5) + '.npy',
                                fused_image[j5, :, :, :])
                # result_diff = (np.mean(np.abs(fused_image[j5, :, :, :] - test_label[j5, :, :, :]), dim=0, keepdims=True))
                # result_diff = np.expand_dims(np.abs(fused_image[index, band_idx, :, :] - test_label[index, band_idx, :, :]), dim=0)
                # np.save(save_dir[j5] + 'Pf_result_diff' + str(j5) + '.npy',
                #             result_diff)
                np.save(save_dir[j5] + 'ms_'+ name + str(j5) + '.npy',
                                F.upsample(torch.tensor(test_ms_image[j5, :, :, :][None, ...]), scale_factor=ratio, mode='bicubic').numpy()[0, ...])
                np.save(save_dir[j5] + 'pan_'+ name + str(j5) + '.npy',
                                test_pan_image[j5, :, :, :])
        noref_results_all = []
        for i5 in range(test_ms_image.shape[0]):
            # temp_ref_results = ref_evaluate(transpose_banddim(fused_image[i5, :, :, :], band=0),
                                            # transpose_banddim(test_label[i5, :, :, :], band=0), scale=ratio)
            temp_noref_results = no_ref_evaluate(np.transpose(fused_image[i5, :, :, :], [1, 2, 0]),
                                            np.transpose(test_pan_image[i5, :, :, :], [1, 2, 0]),
                                            np.transpose(test_ms_image[i5, :, :, :], [1, 2, 0]), scale=ratio)

            # ref_results_all.append(np.expand_dims(temp_ref_results, dim=0))
            noref_results_all.append(np.expand_dims(temp_noref_results, dim=0))

        # ref_results_all = np.concatenate(ref_results_all, dim=0)
        noref_results_all = np.concatenate(noref_results_all, dim=0)
        # ref_results.update({'our       ': np.mean(ref_results_all, dim=0)})
        no_ref_results.update({'our       ': np.mean(noref_results_all, dim=0)})

        if save_images:
            for j5 in range(save_num):
                np.save(save_dir[j5] + 'our_result_'+ name + str(j5) + '.npy',
                                fused_image[j5, :, :, :])
                np.save(save_dir[j5] + 'ms_'+ name + str(j5) + '.npy',
                                F.upsample(torch.tensor(test_ms_image[j5, :, :, :][None, ...]), scale_factor=ratio, mode='bicubic').numpy()[0, ...])
                np.save(save_dir[j5] + 'pan_'+ name + str(j5) + '.npy',
                                test_pan_image[j5, :, :, :])

        metrics_result_no_ref.append(np.mean(noref_results_all, dim=0, keepdims=True))


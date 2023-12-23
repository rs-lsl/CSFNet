
import numpy as np
import pandas as pd
import os
import time
# import cv2
import torch

from metrics import ref_evaluate
from metrics import no_ref_evaluate

from ours.ournet_815_red import ournet

if __name__ == '__main__':

    train_wv2 = False
    train_gf2 = True
    if train_wv2:
        # band_ms = [1, 2, 3, 4, 5]   # wv2
        from save_image_wv2_red import generate_data, crop_data
        name = 'wv2'
        train_num = 110
        test_num = 10
        num_epochs = 500
    elif train_gf2:
        # band_ms = [0, 1, 2, 3]
        from save_image_gf2_red import generate_data, crop_data
        name = 'gf2'
        train_num = 500
        test_num = 50
        num_epochs = 500
    else:
        print(1)

    index3 = 0

    mid_ch = 32  # endmember nums  30
    batch_size = 10
    learning_rate = 1e-4

    ms, pan, label = generate_data()
    ms_crop, pan_crop, label_crop = crop_data(ms, pan, label)
    # ms_crop = np.random.random([1000,4,16,16])
    # pan_crop = np.random.random([1000,1,64,64])
    # label_crop = np.random.random([1000,4,64,64])
    
    index = np.arange(ms_crop.shape[0])
    np.random.seed(1000)
    np.random.shuffle(index)
    # 随机shuffle
    train_ms_image = ms_crop[index[:train_num], :, :, :]
    train_pan_image = pan_crop[index[:train_num], :, :, :]
    train_label = label_crop[index[:train_num], :, :, :]

    # index2 = 45
    test_ms_image = ms_crop[index[-test_num:], :, :, :]
    test_pan_image = pan_crop[index[-test_num:], :, :, :]
    test_label = label_crop[index[-test_num:], :, :, :]

    print(train_ms_image.shape)
    print(train_pan_image.shape)
    print(train_label.shape)

    print(test_ms_image.shape)
    print(test_pan_image.shape)
    print(test_label.shape)

    print(np.max(test_ms_image))
    print(np.max(test_pan_image))
    print(np.max(test_label))

    ratio = int(test_pan_image.shape[2] / test_ms_image.shape[2])
    print('ratio: ', ratio)

    '''setting save parameters'''
    save_num = 5 if name == 'gf2' else 10  # 存储测试影像数目
    save_images = True
    # save_channels = [0, 1, 3]  # BGR for worldview2 image
    save_dir = []
    for i7 in range(save_num):
        save_dir.append('/home/aistudio/result/result_pansharpening_red_'+name+'/results' + str(i7) + '/')
        if save_images and (not os.path.isdir(save_dir[i7])):
            os.makedirs(save_dir[i7])

    '''定义度量指标和度量函数'''
    ref_results = {}
    ref_results.update({'metrics: ': '  SAM,    ERGAS,    Q,    RMSE'})  # 记得更新下面数组长度
    # no_ref_results = {}
    # no_ref_results.update({'metrics: ': '  D_lamda, D_s,    QNR'})
    len_ref_metrics = 7
    # len_no_ref_metrics = 3

    result = []
    result_diff = []
    metrics_result_ref = []  # 存储测试影像指标
    metrics_result_noref = []  # 存储测试影像指标

    test_ournet = True

    '''Pfnet method'''
    if test_ournet:
        print('evaluating ournet method')
        fused_image = ournet(
            train_ms_image, train_pan_image, train_label,
            test_ms_image, test_pan_image, test_label,
            num_epochs=num_epochs, mid_ch=mid_ch, learning_rate=learning_rate,
            batch_size=batch_size, ratio=ratio, name=name)

        fused_image = fused_image.numpy()
        ref_results_all = []
        # noref_results_all = []
        for i5 in range(test_ms_image.shape[0]):
            temp_ref_results = ref_evaluate(np.transpose(fused_image[i5, :, :, :], [1, 2, 0]),
                                            np.transpose(test_label[i5, :, :, :], [1, 2, 0]), scale=ratio)
            # temp_noref_results = no_ref_evaluate(transpose_banddim(fused_image[i5, :, :, :], band=0),
            #                                      transpose_banddim(test_pan_image[i5, :, :, :], band=0),
            #                                      transpose_banddim(test_ms_image[i5, :, :, :], band=0), scale=ratio)

            ref_results_all.append(np.expand_dims(temp_ref_results, dim=0))
            # noref_results_all.append(np.expand_dims(temp_noref_results, dim=0))

        ref_results_all = np.concatenate(ref_results_all, dim=0)
        # noref_results_all = np.concatenate(noref_results_all, dim=0)
        ref_results.update({'our       ': np.mean(ref_results_all, dim=0)})
        # no_ref_results.update({'our       ': np.mean(noref_results_all, dim=0)})
        # print(np.max(test_ms_image))
        if save_images:
            for j5 in range(save_num):
                np.save(save_dir[j5] + 'our_result_'+ name + str(j5) + '.npy',
                        fused_image[j5, :, :, :])
                result_diff = (np.mean(np.abs(fused_image[j5, :, :, :] - test_label[j5, :, :, :]), dim=0, keepdims=True))
                np.save(save_dir[j5] + 'our_result_diff_'+ name + str(j5) + '.npy',
                            result_diff)

                np.save(save_dir[j5] + 'label_'+ name + str(j5) + '.npy',
                        test_label[j5, :, :, :])

        metrics_result_ref.append(np.mean(ref_results_all, dim=0, keepdims=True))


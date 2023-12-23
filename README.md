# CSFNet
The code of the paper **Cross Spectral and Spatial Scale Non-local Attention-Based Unsupervised Pansharpening Network** that has been published in Jstar (https://ieeexplore.ieee.org/document/10130294).

**We have open source the gaofen2 datasets used in this paper to contribute the pansharpening field.** Include the multispectral and pan images, which could be freely downloaded from the website:**https://aistudio.baidu.com/datasetdetail/136740/0**

Note that this dataset is acquired by the Gaofen2 satellite.

You could run the following codes to run the fusion network:

for the full resolution fusion:
```python main_pansharpening_full.py```

for the reduced resolution fusion:
```python main_pansharpening_red.py```

And remember to change the original datasets path in the save_image_gf2.py (for the full resolution gaofen2 dataset):

``` hs_path = '/home/aistudio/data/data136740/msi.npy'  # change path```

``` pan_path = '/home/aistudio/data/data136740/pan.npy' # change path```

In addition, please adjust the epoch for the different datasets, which could achieve better performance.

If you have any questions, please feel free to contact me.

Please consider cite the paper if you find it helpful.

Email:whu_lsl@whu.edu.cn



from torch.utils.data import Dataset
# import numpy as np
# from PIL import Image
# from scipy.misc import imresize
# from scipy.ndimage.interpolation import rotate

class Mydata_test(Dataset):
    def __init__(self, lrhs, pan):
        super(Mydata_test, self).__init__()
        self.lrhs = lrhs
        self.pan = pan

    def __getitem__(self, idx):
        assert idx < self.pan.shape[0]

        return self.lrhs[idx, :, :, :], self.pan[idx, :, :, :]  # , self.label[idx, :, :, :]

    def __len__(self):
        return self.pan.shape[0]

class Mydata(Dataset):
    def __init__(self, lrhs, pan):
        super(Mydata, self).__init__()
        self.lrhs = lrhs
        self.pan = pan

    def __getitem__(self, idx):
        assert idx < self.pan.shape[0]

        return self.lrhs[idx, :, :, :], self.pan[idx, :, :, :]  # , self.label[idx, :, :, :]

    def __len__(self):
        return self.pan.shape[0]

class Mydata2(Dataset):
    def __init__(self, lrhs, pan, label):
        super(Mydata2, self).__init__()
        self.lrhs = lrhs
        self.pan = pan
        self.label = label

    def __getitem__(self, idx):
        assert idx < self.pan.shape[0]

        return self.lrhs[idx, :, :, :], self.pan[idx, :, :, :], self.label[idx, :, :, :]

    def __len__(self):
        return self.pan.shape[0]


# class Mydata(Dataset):
#     def __init__(self, lrhs, pan):
#         super(Mydata, self).__init__()
#         self.lrhs = lrhs
#         self.pan = pan

#     def __getitem__(self, idx):
#         assert idx < self.pan.shape[0]

#         return self.data_augmentation(self.lrhs[idx, :, :, :], self.pan[idx, :, :, :])  # , self.label[idx, :, :, :]
#         # self.spec_graident_weight, self.pan_weightfactor

#     def data_augmentation(self, ms, pan):
#         ms, pan = horizontal_flip(ms, pan, rate=0.5)
#         ms, pan = vertical_flip(ms, pan, rate=0.5)
#         ms, pan = random_rotation(ms, pan, rate=1)
#         # print(ms.shape)
#         # print(pan.shape)
#         return ms, pan

#     def __len__(self):
#         return self.pan.shape[0]

# def horizontal_flip(image, image2, rate=0.5):
#     if np.random.rand() < rate:
#         image = image[:, ::-1, :]
#         image2 = image2[:, ::-1, :]
#     return image, image2

# def vertical_flip(image, image2, rate=0.5):
#     if np.random.rand() < rate:
#         image = image[:, :, ::-1]
#         image2 = image2[:, :, ::-1]
#     return image, image2

# def random_rotation(image, image2, angle_range=(0, 360), ratio=4, rate=0.2):
#     if np.random.rand() < rate:
#         image = image.permute(1,2,0])
#         image2 = image2.permute(1,2,0])

#         rotation_angle = [90, 180, 270]
#         h, w, _ = image.shape
#         angle = rotation_angle[np.random.randint(0, 3)]

#         image = rotate(image, angle)
#         image2 = rotate(image2, angle)
#         # image = resize(image, (h, w))
#         # image2 = resize(image2, (h*ratio, w*ratio))
#         # print(image.permute(2,0,1]).shape)
#         # print(image2.permute(2,0,1]).shape)
#     return image.permute(2,0,1]), image2.permute(2,0,1])

# def resize(image, size):
#     size = check_size(size)
#     # image = imresize(image, size)
#     image = np.array(Image.fromarray(image).resize(size))
#     return image

# def check_size(size):
#     if type(size) == int:
#         size = (size, size)
#     if type(size) != tuple:
#         raise TypeError('size is int or tuple')
#     return size





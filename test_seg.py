import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import imageio
import os
from torch.autograd import Variable
from torch.utils import data
import cv2

from skimage.transform import resize
from ptsemseg.models.segnet import *
from ptsemseg.models.pspnet import *
import collections

def convert(dic):
    res = collections.OrderedDict()
    for k, v in dic.items():
        name = k[7:] # remove `module.`
        res[name] = v
#         del dic[k]
    return res

class SUNRGBDTESTLoader(data.Dataset):
    def __init__(self, rootname, folderNameList= None, tailNames= None, is_transform = False, img_size=(240,320)):
        self.root = rootname

        self.n_classes = 13

        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([125.91207973, 116.5486107 , 110.43807554, 132.84413787,  81.09253009,  93.7494152])
        self.datasize = len(os.listdir(self.root + 'SUNRGBD-train_images/'))

        self.folders = ['rgb/', 'hha/'] if folderNameList == None else folderNameList
        self.tailnames  = ['.png','.png'] if tailNames == None else tailNames
        # label:
        self.lableColor = {1:[173,216,230], 2:[255,0,0], 3:[224,255,255], 4:[165,42,42], 5:[238,216,174],\
        6:[255,165,0], 7:[173,255,47],8:[132,112,255],9:[147,112,219],10:[139,69,0],11:[255,106,106],12:[255,218,185],13:[255,240,245]}
        # 1	Bed # 2	Books  # 3	Ceiling # 4	Chair# 5	Floor# 6	Furniture# 7	Objects# 8	Picture# 9	Sofa# 10	Table# 11	TV# 12	Wall# 13	Window
    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        index = index +1
        rgb_path = self.root + self.folders[0] + str(index) + self.tailnames[0]
        hha_path = self.root + self.folders[1] + str(index) + self.tailnames[1]

        rgb = imageio.imread(rgb_path)
        rgb = np.array(rgb, dtype = np.uint8)
        hha = imageio.imread(hha_path)
        hha = np.array(hha, dtype = np.uint8)

        img = np.concatenate((rgb,hha), axis = 2)

        if(self.transform):
            img = self.transform(img)
        return img

    def transform(self, img):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = resize(img, (self.img_size[0], self.img_size[1]), mode='reflect')
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()

        return img

    def decode_segmap(self, coded, plot=True):
        r = coded.copy()
        g = coded.copy()
        b = coded.copy()
        rgb = np.zeros((coded.shape[0], coded.shape[1],3), dtype=np.uint8)
        foundClass = np.unique(coded)
        for label in foundClass:
            if (label in self.lableColor):
                color = self.lableColor[label]
            else:
                color = [255,0,0]
            idxMat = (coded == label)
            r[idxMat] = color[0]
            g[idxMat] = color[1]
            b[idxMat] = color[2]
        rgb[:,:,0] = r
        rgb[:,:,1] = g
        rgb[:,:,2] = b
        if plot:
            cv2.imshow("inside",rgb)
        return rgb

data_path = "C:/Projects/SUNRGB-dataset/"
folderList = ['SUNRGBD-test_images/', 'testing/hha/']
tailTypes =  ['.jpg','.png']
testdata = SUNRGBDTESTLoader(data_path, folderList, tailTypes, is_transform=True)
testloader = torch.utils.data.DataLoader(testdata, batch_size = 1, shuffle = False)
# Setup Model
model = pspnet(n_classes=testdata.n_classes, in_channels=6)
# model = segnet(n_classes=testdata.n_classes,is_unpooling=True, in_channels=6)
# state = convert_state_dict(torch.load("segnet_sunrgbd_sun_model.pkl")['model_state'])
state = torch.load("pspnet_sunrgbd_sun_model.pkl")['model_state']
state = convert(state)
model.load_state_dict(state)
model.eval()

model.cuda(0)
for i, images in enumerate(testloader):
    images = Variable(images.cuda(0), volatile=True)
    outputs = F.softmax(model(images), dim=1)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = testdata.decode_segmap(pred)
    print('Classes found: ', np.unique(pred))
    # print(decoded.shape)
    # colored = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    print(decoded[:10,:10,:])
    cv2.imshow("test",decoded)
    cv2.waitKey(0)
    # imageio.imwrite("test_out_" + str(i)+".png", decoded)

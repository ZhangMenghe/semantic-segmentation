import os
import collections
import imageio
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.models.segnet import *

class NYUD2Loader(data.Dataset):
    def __init__(self, root, split= "training", is_transform = False, img_size=(240,320), splitRate = 0.7):
        self.root = root + 'imgs/'
        self.n_classes = 40
        self.split = split
        self.splitRate = splitRate
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([122.5454, 104.7834, 100.0239,134.5181,110.9748,137.2213])
        self.files_rgb = recursive_glob(rootdir=self.root + 'rgb/', suffix='.png')
        self.datasize = len(self.files_rgb)
        self.startIndex = 0 if(split=="training") else int(self.datasize*splitRate)

    def __len__(self):
        if(self.split == "training"):
            return int(self.datasize*self.splitRate)
        return int(self.datasize*(1-self.splitRate))+1

    def __getitem__(self, index):
        index = index + self.startIndex+1
        rgb_path = self.root + 'rgb/' + str(index) + '.png'
        hha_path = self.root + 'hha/' + str(index) + '.png'
        lbl_path = self.root + 'label/' + str(index) + '.png'

        rgb = imageio.imread(rgb_path)
        rgb = np.array(rgb, dtype = np.uint8)
        hha = imageio.imread(hha_path)
        hha = np.array(hha, dtype = np.uint8)
        lbl = imageio.imread(lbl_path)
        lbl = np.array(lbl, dtype = np.int32)

        img = np.concatenate((rgb,hha), axis = 2)

        if(self.transform):
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = resize(img, (self.img_size[0], self.img_size[1]), mode='reflect')
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = resize(lbl, (self.img_size[0], self.img_size[1]), mode='reflect')
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
mbatch_size = 4
mn_epoch = 10
mvisdom = False
march = 'segnet'
mdataset = 'nyud2'
ml_rate = 0.1
mresume = None
data_path = "C:/Projects/getHHA/"
traindata = NYUD2Loader(data_path, split='training', is_transform = True)
trainloader = torch.utils.data.DataLoader(traindata, batch_size = mbatch_size, shuffle=True)

valdata = NYUD2Loader(data_path, split='validation', is_transform = True)
valloader = torch.utils.data.DataLoader(valdata, batch_size= mbatch_size, shuffle=True)

# n_classes = traindata.n_classes


model = segnet(n_classes=traindata.n_classes,is_unpooling=True, in_channels=6)
# vgg16 = models.vgg16(pretrained=True)
# model.init_vgg16_params(vgg16)
model = nn.DataParallel(model, device_ids= range(torch.cuda.device_count()))
model.cuda()

if hasattr(model.module, 'optimizer'):
    optimizer = model.module.optimizer
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=ml_rate, momentum=0.99, weight_decay=5e-4)

if hasattr(model.module, 'loss'):
    print('Using custom loss')
    loss_fn = model.module.loss
else:
    loss_fn = cross_entropy2d
# Setup Metrics
running_metrics = runningScore(traindata.n_classes)

best_iou = -100.0
for epoch in range(mn_epoch):
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(input=outputs, target=labels)

        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, mn_epoch, loss.data[0]))

    model.eval()
    for i_val, (images_val, labels_val) in enumerate(valloader):
        images_val = Variable(images_val.cuda(), volatile=True)
        labels_val = Variable(labels_val.cuda(), volatile=True)

        outputs = model(images_val)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    running_metrics.reset()

    if score['Mean IoU : \t'] >= best_iou:
        best_iou = score['Mean IoU : \t']
        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 'optimizer_state' : optimizer.state_dict(),}
        torch.save(state, "{}_{}_best_model.pkl".format(march, mdataset))

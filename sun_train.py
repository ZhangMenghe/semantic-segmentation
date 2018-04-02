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
from ptsemseg.models.pspnet import *
class SUNRGBDLoader(data.Dataset):
	def __init__(self, rootname, folderNameList= None, tailNames= None, split = "training", is_transform = False, img_size=(240,320), splitRate = 0.7):
		self.root = rootname
		self.split = split
		self.n_classes = 12
		self.splitRate = splitRate
		self.is_transform = is_transform
		self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
		self.mean = np.array([125.74308659806684, 116.25128126375019, 110.39211491755876, 134.51503601619351, 79.703766694807911, 92.11112833151347])
		# self.mean = np.array([125.91207973, 116.5486107 , 110.43807554])
		# self.files_rgb = recursive_glob(rootdir=self.root + 'SUNRGBD-train_images/', suffix='.jpg')
		fullnames = os.listdir(rootname+folderNameList[0])
		self.datasize = len(fullnames)
		self.imgNameSet = [fullnames[i].split('.')[0] for i in range(self.datasize)]
		self.startIndex = 0 if(split=="training") else int(self.datasize*splitRate)
		self.folders = ['rgb/', 'hha/', 'labels'] if folderNameList == None else folderNameList
		self.tailnames  = ['.png','.png','.png'] if tailNames == None else tailNames


	def __len__(self):
		if(self.split == "training"):
			return int(self.datasize*self.splitRate)
		return int(self.datasize*(1-self.splitRate))+1

	def __getitem__(self, index):
		index = index + self.startIndex
		rgb_path = self.root + self.folders[0] + self.imgNameSet[index] + self.tailnames[0]
		hha_path = self.root + self.folders[1] + self.imgNameSet[index] + self.tailnames[1]
		lbl_path = self.root + self.folders[2] + self.imgNameSet[index] + self.tailnames[2]

		rgb = imageio.imread(rgb_path)
		rgb = np.array(rgb, dtype = np.uint8)
		hha = imageio.imread(hha_path)
		hha = np.array(hha, dtype = np.uint8)
		lbl = imageio.imread(lbl_path)
		lbl = np.array(lbl, dtype = np.int32)

		img = np.concatenate((rgb,hha), axis = 2)
		# img = rgb
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
mn_epoch = 400
mvisdom = False
march = 'pspnet'
mdataset = 'sunrgbd'
ml_rate = 0.1
mresume = None
data_path = "C:/Projects/SUNRGB-dataset/_training/"
loss_file = data_path+"loss.txt"
# folderList = ['SUNRGBD-train_images/', 'hha/','train13labels/']
folderList = ['imgs/', 'hha/','label12/']
tailTypes =  ['.jpg','.png','.png']
start_epoch = 0
resume = False
resume_root = "pspnet_sunrgbd_sun_model_resume.pkl"
traindata = SUNRGBDLoader(data_path, folderList, tailTypes, split='training', is_transform = True)
trainloader = torch.utils.data.DataLoader(traindata, batch_size = mbatch_size, shuffle=True)

valdata = SUNRGBDLoader(data_path, folderList, tailTypes,  split='validation', is_transform = True)
valloader = torch.utils.data.DataLoader(valdata, batch_size= mbatch_size, shuffle=True)

# n_classes = traindata.n_classes


# model = segnet(n_classes=traindata.n_classes,is_unpooling=True, in_channels=3)
model = pspnet(n_classes=traindata.n_classes, in_channels=6)

model = nn.DataParallel(model, device_ids= range(torch.cuda.device_count()))
model.cuda()

if hasattr(model.module, 'optimizer'):
    optimizer = model.module.optimizer
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=ml_rate, momentum=0.9, weight_decay=5e-4)
if(resume):
	if(os.path.isfile(resume_root)):
		print("==> Loading Half-training model...'{}'".format(march))
		checkpoints = torch.load(resume_root)
		start_epoch = checkpoints['epoch']
		state = checkpoints['model_state']
		model.load_state_dict(state)
		optimizer.load_state_dict(checkpoints['optimizer_state'])
		print("==> Checkpoint '{}' Loaded, start from epoch {}".format(march,checkpoints["epoch"]))

if hasattr(model.module, 'loss'):
    print('Using custom loss')
    loss_fn = model.module.loss
else:
    loss_fn = cross_entropy2d
# Setup Metrics
running_metrics = runningScore(traindata.n_classes)

best_iou = -100.0
for epoch in range(start_epoch, mn_epoch):
	if(epoch %100 == 0):
	    ml_rate = ml_rate * 0.1
	    optimizer = torch.optim.SGD(model.parameters(), lr=ml_rate, momentum=0.9, weight_decay=5e-4)
	storeLoss =[]
	model.train()
	# print(len(trainloader))
	for i, (images, labels) in enumerate(trainloader):
		images = Variable(images.cuda())
		labels = Variable(labels.cuda())
		optimizer.zero_grad()
		outputs = model(images)

		loss = loss_fn(input=outputs, target=labels)

		loss.backward()
		optimizer.step()
		storeLoss.append(loss.data[0])
		if (i+1) % 20 == 0:
		    print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, mn_epoch, loss.data[0]))

	fp = open(loss_file,'a+')
	fp.write("Epoch [%d/%d]\n" % (epoch+1, mn_epoch))
	fp.write('\t'.join(storeLoss) + '\n')
	fp.close()

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
	    torch.save(state, "{}_{}_sun_model2_resume.pkl".format(march, mdataset))

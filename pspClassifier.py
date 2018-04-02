import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from ptsemseg.models.pspnet import *
import collections
from SUNRGBDTESTLoader import *
import numpy as np

class pspClassifier(object):
    """docstring for pspClassifier."""
    def __init__(self, root_path, srcImgPath, tailTypes = ['.jpg','.png'], modelFile="pspnet_sunrgbd_sun_model.pkl"):
        self.dataset = SUNRGBDTESTLoader(root_path, srcImgPath, is_transform=False)

        # Setup Model
        self.model = pspnet(n_classes=self.dataset.n_classes, in_channels=6)
        self.state = torch.load(modelFile)['model_state']
        self.convert()
        self.model.load_state_dict(self.state)
        self.model.eval()
        self.model.cuda(0)
    def convert(self):
        res = collections.OrderedDict()
        for k, v in self.state.items():
            name = k[7:] # remove `module.`
            res[name] = v
        self.state =  res
    def fit(self, imageName, hha):
        rgb = self.dataset[imageName]
        img = np.concatenate((rgb,hha), axis = 2)
        img = self.dataset.transform(img)
        torchLoader = torch.utils.data.DataLoader([img], batch_size = 1, shuffle = False)
        for image in torchLoader:
            image = Variable(image.cuda(0), volatile=True)
            outputs = F.softmax(self.model(image), dim=1)
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            print('Classes found: ', np.unique(pred))
            return pred
    def fitList(self, imgList):
        res = []
        for img in imgList:
            res.append(self.fit(img))
        return res

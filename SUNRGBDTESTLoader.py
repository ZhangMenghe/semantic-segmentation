import torch
from torch.utils import data
import numpy as np
import os
import imageio
from skimage.transform import resize
class SUNRGBDTESTLoader(data.Dataset):
    def __init__(self, rootname, srcImgPath= None, is_transform = False, img_size=(240,320)):
        self.root = rootname
        self.n_classes = 12
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([125.74308659806684, 116.25128126375019, 110.39211491755876, 134.51503601619351, 79.703766694807911, 92.11112833151347])
        self.datasize = len(os.listdir(srcImgPath))
        self.rgb_path = srcImgPath
        #self.cate = ["unknown", "floor ","sofa ","chair ","bed ","NightStand","shelf","table","wall","onwallObjs","otherFurniture","ceiling"]
        self.labelColor = {1:[173,216,230], 2:[139, 0 ,139], 3:[255,0,0], 4:[156, 156, 156], 5:[0,255,0],\
        6:[255,165,0], 7:[173,255,47],8:[255, 228, 225],9:[159, 121, 238],10:[139,69,0],11:[255,106,106],12:[0,0,255],13:[255,2552,255]}

    def __len__(self):
        return self.datasize

    def __getitem__(self, imageName):
        rgb = imageio.imread(self.rgb_path + imageName)
        return rgb

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

    def decode_segmap(self, coded, plot=False):
        r = coded.copy()
        g = coded.copy()
        b = coded.copy()
        rgb = np.zeros((coded.shape[0], coded.shape[1],3), dtype=np.uint8)
        foundClass = np.unique(coded)
        for label in foundClass:
            if (label in self.labelColor):
                color = self.labelColor[label]
            else:
                color = [0,0,0]
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
    def writeColorRef(self):
        labelMap = np.zeros((300,400,3), np.uint8)
        for i in range(3):
            for j in range(4):
                r,g,b = self.labelColor[4*i+j + 1]
                labelMap[100*i:100*(i+1), 80*j:80*(j+1)] = [b,g,r]
        labelMap[200:300, 320:400]  = [255,255,255]
        cv2.imwrite("labelMap.png", labelMap)

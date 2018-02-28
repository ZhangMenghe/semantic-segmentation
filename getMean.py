import os
import numpy as np
import imageio

def getMeanValue():
    rgb_path = "C:/Projects/SUNRGB-dataset/SUNRGBD-train_images/"
    hha_path = "C:/Projects/SUNRGB-dataset/hha/"
    path_set = [rgb_path, hha_path]
    result = []
    for path in path_set:
        files = os.listdir(path)
        numOfFiles = len(files)
        mean_record = np.zeros(3)
        for file in files:
            img = np.array(imageio.imread(path + file))
            mean_record[0] += np.mean(img[:,:,0])
            mean_record[1] += np.mean(img[:,:,1])
            mean_record[2] += np.mean(img[:,:,2])
        result.extend(mean_record/numOfFiles)
    return result

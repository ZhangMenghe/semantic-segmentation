from pspClassifier import *
import cv2
data_path = "C:/Projects/SUNRGB-dataset/_training/"
folderList = ['imgs/', 'hha/']
# folderList = ['SUNRGBD-test_images/', 'testing/hha/']
tailTypes =  ['.jpg','.png']
labelName  = 'label12/'
classifier = pspClassifier(data_path, data_path+folderList[0],modelFile ="pspnet_sunrgbd_sun_model2_resume.pkl")
# testList=np.array([1923,1935,2021,2129,2163,2210,2214,2250,2264])
testList=np.array([2496])
numOfTest = max(testList)
for i in testList:
    ori = cv2.imread(data_path+folderList[0]+str(i)+'.jpg')
    label = cv2.imread(data_path+labelName+str(i)+'.png', 0)
    hha = imageio.imread(data_path+folderList[1]+str(i)+'.png')
    decoded_label = classifier.dataset.decode_segmap(label)
    pred = classifier.fit(str( i)+'.jpg',hha)
    decoded = classifier.dataset.decode_segmap(pred)
    decoded = decoded.astype(np.uint8)
    cv2.imshow("prediction",decoded)
    cv2.imshow("GroundTruth", decoded_label)
    cv2.waitKey(0)

    # cv2.imwrite("E:/ori-" + str(i+1)+'.jpg', ori)
    # cv2.imwrite("E:/ground-" + str(i+1)+'.png', decoded_label)
    # cv2.imwrite("E:/pred-" + str(i+1)+'.png', decoded)

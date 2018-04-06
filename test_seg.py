from pspClassifier import *
import cv2
data_path = "C:/Projects/SUNRGB-dataset/_training/"
folderList = ['imgs/', 'hha/']
# folderList = ['SUNRGBD-test_images/', 'testing/hha/']
tailTypes =  ['.jpg','.png']
labelName  = 'label12/'
classifier = pspClassifier(data_path, data_path+folderList[0],modelFile ="pspnet_sunrgbd_sun_model2_resume.pkl")
testList=np.array([1923,1935,2021,2129,2163,2210,2214,2250,2264])
numOfTest = max(testList)
for i in testList:
    ori = cv2.imread(data_path+folderList[0]+str(i+1)+'.jpg')
    label = cv2.imread(data_path+labelName+str(i+1)+'.png', 0)
    hha = imageio.imread(data_path+folderList[1]+str(i+1)+'.png')
    decoded_label = classifier.dataset.decode_segmap(label)
    pred = classifier.fit(str( i+1)+'.jpg',hha)
    # mat = np.zeros(label.shape)
    # mat[np.where(pred == 5)] = 1
    # cv2.imshow("test1", mat)
    # cv2.waitKey(0)
    decoded = classifier.dataset.decode_segmap(pred)
    decoded = decoded.astype(np.uint8)
    # cv2.imshow("test",decoded_label)
    # cv2.waitKey(0)
    cv2.imwrite("E:/ori-" + str(i+1)+'.jpg', ori)
    cv2.imwrite("E:/ground-" + str(i+1)+'.png', decoded_label)
    cv2.imwrite("E:/pred-" + str(i+1)+'.png', decoded)
# for i, images in enumerate(testloader):
#     idx = i+1
#     # if(idx>numOfTest):
#     #     break
#     # if(idx not in testList):
#     #     continue
#     images = Variable(images.cuda(0), volatile=True)
#     print(images.shape)
#     outputs = F.softmax(model(images), dim=1)
#     pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
#     # decoded = testdata.decode_segmap(pred)
#     print('Classes found: ', np.unique(pred))
#     # print(decoded.shape)
#     # colored = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
#     # cv2.imshow("test",decoded)
#     # cv2.waitKey(0)
#     cv2.imwrite("pred" + str(i+1)+".png", pred)

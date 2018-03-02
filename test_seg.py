from pspClassifier import *

data_path = "C:/Projects/SUNRGB-dataset/"
folderList = ['SUNRGBD-test_images/', 'testing/hha/']
tailTypes =  ['.jpg','.png']
classifier = pspClassifier(data_path, folderList)
pred = classifier.fit("1")
print('MIE!!!!Classes found: ', np.unique(pred))

# testList=np.array([1970,1972,1975,2115,2243,2291,2293,2295,2297,2300,2321,2322,2330,2342,2348,2349,2352,2354,2377,2411,2441,2490])
# numOfTest = max(testList)
#
#
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

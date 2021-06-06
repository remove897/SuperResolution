# encoding: utf-8
import matplotlib
matplotlib.use('TkAgg')

import cv2
import os
import matplotlib.pyplot as plt

from skimage.metrics import  peak_signal_noise_ratio
from skimage.metrics import  structural_similarity
from skimage.metrics import  mean_squared_error

# skimage.io.imread((img_path)) 读出来是numpy，(height,width,channel)
# cv2.imread(img_path) 读进来的图片已经是一个numpy矩阵了！！！彩色图片维度是(高度，宽度，通道数)。数据类型是uint8

def loadImagesFromFolder(folder):
    images=[]
    for filename in os.listdir(folder):
        path = os.path.join(folder,filename)
        if os.path.isfile(path) and filename.startswith('test'):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if (img is not None):
                images.append(img)
    return images

def toGray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


origin_path = '/Users/didi/Downloads/PNG T1/PNG T1/T1_074/T1_074_0088.png'
target_images = '/Users/didi/Desktop/result'
origin = cv2.imread(origin_path)
origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
targets = loadImagesFromFolder(target_images)
#预测图像一般是1024*1024的，注意resize（1024，2048）的shape是（2048，1024）
origin  = cv2.resize(origin,(1024,1024),interpolation=cv2.INTER_AREA)

for i in range(0,len(targets)):
    plt.imshow(toGray(targets[i]))
    plt.show()
    mse = mean_squared_error(origin,targets[i])
    print("MSE:{}".format(mse))
    psnr = peak_signal_noise_ratio(origin,targets[i])
    print("PSNR:{}".format(psnr))
    ssim = structural_similarity(toGray(origin), toGray(targets[i]))
    print("SSIM:{}".format(ssim))
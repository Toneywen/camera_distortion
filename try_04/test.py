import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((366,487,channels),np.uint8)
    sh=366/height
    sw=487/width
    for i in range(366):
        for j in range(487):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage
 
img=cv2.imread("/home/wenxiangyu/project/camera_distortion/normal_7.jpg")
zoom=function(img)
# cv2.imshow("nearest neighbor",zoom)
# cv2.imshow("image",img)
# cv2.waitKey(0)
# coding: utf-8

import numpy as np
import torch
import cv2 
import math
import time

def distortImg(srcImg, dstImg, fx, fy, cx, cy):
    imgHeight, imgWidth = srcImg.shape[:2]

    D = [0.3782, 0.9719, 0, 0, -2.9170]
    pSrcData = srcImg
    value = [0,0,0]
    start = time.time()

    for j in range(imgHeight):
        for i in range(imgWidth):
            # //转到摄像机坐标系
            X = (i-cx) / fx
            Y = (j-cy) / fy
            r2 = X * X + Y * Y 
            # //加上畸变
            newX = X * (1 + D[0]*r2 + D[1]*r2*r2)
            newY = Y * (1 + D[0]*r2 + D[1]*r2*r2)

            # //再转到图像坐标系
            u = newX * fx + cx
            v = newY * fy + cy
            # //双线性插值
            u0 = math.floor(u)
            v0 = math.floor(v)
            u1 = u0 + 1
            v1 = v0 + 1

            dx = u - u0
            dy = v - v0
            weight1 = (1-dx) * (1-dy)
            weight2 = dx * (1-dy)
            weight3 = (1-dx) * dy
            weight4 = dx * dy

            for k in range(3):
                if u0 >= 0 and u1 < imgWidth and v0 >= 0 and v1 < imgHeight:
                    value[k]=int(pSrcData[j,i][k]*weight1+pSrcData[j,i+1][k]*weight2+pSrcData[j+1,i][k]*weight3+pSrcData[j+1,i+1][k]*weight4)
                else:
                    value = [0,0,0]

            dstImg[j, i] = (value[0], value[1], value[2])

            # # 最近临插值
            # u = round(u)
            # v = round(v)
            # if (u0 >= 0 and v0 >= 0 and u1 < imgWidth and v1 < imgHeight):
            #     dstImg[j, i] = pSrcData[v1, u1]
            #     # dstImg[j, i] = pSrcData[math.floor(v), math.floor(u)]
            # else:
            #     dstImg[j, i] = 0
            
    end = time.time()
    print('using time: ', end - start)
    return dstImg

if __name__ == '__main__':
    imgPath="/home/wenxiangyu/project/camera_distortion/normal_7.jpg"

    srcImg = cv2.imread(imgPath)
    print(type(srcImg))
    # dstImg = np.zeros(srcImg.shape[:2])
    dstImg = srcImg

    rate_ = 0.6
    fx=982.140
    fy=982.140
    fx = fx * rate_
    fy = fy * rate_
    cx=243.500
    cy=183.000

    dstImg = distortImg(srcImg, dstImg, fx, fy, cx, cy)

    cv2.imwrite('/home/wenxiangyu/dstImg.jpg', dstImg)

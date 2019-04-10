# coding: utf-8
import numpy as np
import scipy
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from interp2 import interp2linear
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

dist_coefs = np.array([ 3.7816712271668235e-01, 9.7194428295751345e-01, 0, 0, -2.9169719384592412e+00])

camera_matrix = np.array([[ 9.8214053893310370e+02    ,0.  ,6.3950000000000000e+02],
                        [   0.          ,9.8214053893310370e+02,  3.5950000000000000e+02],
                        [   0.            ,0.            ,1.        ]])

fx = camera_matrix[0][0]
fy = camera_matrix[1][1]  
cx = camera_matrix[0][2]  
cy = camera_matrix[1][2]  
k1 = dist_coefs[0]  
k2 = dist_coefs[0]    
k3 = dist_coefs[0]    
p1 = dist_coefs[0]    
p2 = dist_coefs[0]

K = camera_matrix

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

Idistorted = np.array(cv.imread('/home/wenxiangyu/project/camera_distortion/normal_2.jpg'))
print(Idistorted.shape)
# Idistorted = rgb2gray(Idistorted) 
Idistorted = cv.resize(Idistorted, (463, 344))
gray = rgb2gray(Idistorted)
# print(gray.shape)

I = np.zeros(gray.shape)
# print(I)
# print(I.shape)
# print(I.shape[0] * I.shape[1])

# i, j = np.where(np.isnan(I))
# i, j = I.shape
i = []
j = []

for i_idx in range(I.shape[0] * I.shape[1]):
    i.append(i_idx)
for j_idx in range(I.shape[1] * I.shape[0]):
    j.append(j_idx)
i = np.array(i).reshape(len(i), 1)
j = np.array(j).reshape(len(j), 1)
# print(np.isnan(I))
# print(I.nonzero())

# Xp = inv(K)*[j i ones(length(i),1)]'; 
# print(len(np.hstack((np.hstack((j, i)),np.ones(len(i), 1)))))

# print(len(np.hstack((np.hstack((j,i)),np.ones(len(i))))))
k_1 = np.hstack((np.hstack((j,i)),(np.ones(len(i))).reshape(len(i), 1)))
# print(len(k_1))
Xp = np.dot(np.linalg.inv(K), np.transpose(k_1))

x = Xp[0,:]  
y = Xp[1,:] 

r2 = x*x + y*y

# x = x.*(1+k1*r2 + k2*r2.^2) + 2*p1.*x.*y + p2*(r2 + 2*x.^2);  
# y = y.*(1+k1*r2 + k2*r2.^2) + 2*p2.*x.*y + p1*(r2 + 2*y.^2);

x = x*(1+k1*r2 + k2*r2*r2) + 2*p1*x*y + p2*(r2+2*x*x)
y = y*(1+k1*r2 + k2*r2*r2) + 2*p2*x*y + p1*(r2+2*y*y)

# u = reshape(fx*x+cx,size(I));  
# v = reshape(fy*y+cy,size(I)); 
u = (fx*x+cx).reshape(I.shape)
v = (fy*y+cy).reshape(I.shape)

print((fx*x+cx).shape)
print(u.shape)

# I = interp2linear(Idistorted, u, v, 'cubic')
I = RectBivariateSpline(Idistorted,u,v)

plt.imshow(I, cmap = plt.get_cmap('gray'))
plt.show()
# cv.imshow('test', gray)

# cv.waitKey(10000)
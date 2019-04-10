# '''
# camera calibration for distorted images with chess board samples
# reads distorted images, calculates the calibration and write undistorted images
 
# usage:
#    calibrate.py [--debug <output path>] [--square_size] [<image mask>]
 
# default values:
#    --debug:    ./output/
#    --square_size: 1.0
#    <image mask> defaults to ../data/left*.jpg
# '''
 
# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2
from PIL import Image  
# local modules
from common import splitfn
# built-in modules
import os
import sys
import getopt
from glob import glob

def distortion(img_PIL): 
    h, w = 0, 0
    
    print('processing ...')

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  

    if img is None:
        print("Failed to load the image from Pillow pattern ...")
        return

    h, w = img.shape[:2]

    # n = 54 
    
    camera_matrix = np.eye(3)
    dist_coefs = np.zeros(4)
    # calib_flags=cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
   
    dist_coefs = np.array([ 3.7816712271668235e-01, 9.7194428295751345e-01, 0, 0, -2.9169719384592412e+00])
    
    # the coefficient matrix to numtiply on the dist_coefs matrix 
    # to see how the Ks change the distortion of one normal image 
    coef_mat_numti = np.array([1, 0, 0, 0, 0])
    dist_coefs = dist_coefs * coef_mat_numti
    camera_matrix = np.array([[ 9.8214053893310370e+02    ,0.  ,6.3950000000000000e+02],
                            [   0.          ,9.8214053893310370e+02,  3.5950000000000000e+02],
                            [   0.            ,0.            ,1.        ]])
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))        

    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    cv2.imwrite('/home/wenxiangyu/change_weight_1.500.png', dst)
    
    print('ok')
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = Image.open('/home/wenxiangyu/project/camera_distortion/normal_2.jpg')  

    distortion(image)
# coding: utf-8 
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
import cupy as cp
import cv2
# local modules
from common import splitfn
# built-in modules
import os
 
if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob
 
    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'hello='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1)
    args.setdefault('--hello', 'hello')
    if not img_mask:
        img_mask = "/home/wenxiangyu/project/camera_distortion/normal_6.jpg"  # default
    else:
        img_mask = img_mask
 
    img_names = glob(img_mask)
    print('img_names',img_names)
    debug_dir = args.get('--debug')

    # for testint the terminal para
    # hello = args.get('--hello')
    # print(hello)
    if not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))
 
    pattern_size = (9, 6)
    pattern_points = cp.zeros((np.prod(pattern_size), 3), cp.float32)
    pattern_points[:, :2] = cp.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    
    # print('pattern_points', pattern_points)

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
   
    for fn in img_names:
        print('processing %s... ' % fn, end='')
        img = cv2.imread(fn, 0)
        # img = cv2.resize(img, (463, 344))
        # cv2.imshow('fdd', img)
        # waitkey()
        if img is None:
            print("Failed to load", fn)
            continue
 
        h, w = img.shape[:2]
        # print('h,w', h, w)
        # found, corners = cv2.findChessboardCorners(img, pattern_size)
        # print(found)
        # if found:
        #     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        #     cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
 
        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            outfile = debug_dir + name + '_chess.png'
            cv2.imwrite(outfile, vis)
            # if found:
            img_names_undistort.append(outfile)
 
        # if not found:
        #     print('chessboard not found')
        #     continue
 
        # img_points.append(corners.reshape(-1, 2))
        # obj_points.append(pattern_points)
 
        print('ok')
 
    # calculate camera distortion    
    n = 54
    # obj_points=np.asarray([obj_points],dtype='float64').reshape(-1,1,n,3)
    # img_points=np.asarray([img_points],dtype='float64').reshape(-1,1,n,2)
    
    # print(img_points)

    camera_matrix = cp.eye(3)
    # print('camera_matrix', camera_matrix)
    dist_coefs = cp.zeros(4)
    calib_flags=cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
   
    # rms, camera_matrix, dist_coefs, rvecs, tvecs=cv2.fisheye.calibrate(obj_points, img_points, (w, h), camera_matrix, dist_coefs, None, None)
   
    # print("nRMS:", rms)
    print("camera matrix:n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
 
    # undistort the image with the calibration
    print('')
    for img_found in img_names_undistort:
        img = cv2.imread(img_found)
        # cv2.imwrite('img_found.jpg', img)
        h, w = img.shape[:2]
        print(h, w)

        # the source data of camera matrix
        # dist_coefs = np.array([ -3.7816712271668235e-01, -9.7194428295751345e-01, 0, 0, 2.9169719384592412e+00])
        
        # the positive data of camera matrix, to distort the normal image 
        # 大长方形
        dist_coefs = cp.array([ 3.7816712271668235e-01, 9.7194428295751345e-01, 0, 0, -2.9169719384592412e+00])
        # 大正方形
        # dist_coefs = np.array([-5.7609654518727027e-01, -7.0720161054442832e+00, 0., 0., 1.3512744755611772e+02])
        # 小正方形
        # dist_coefs = np.array([1.8749995414564089e+00, -8.1959543498061620e+01, 0., 0., 6.2033712003569826e+02])
        
        # the coefficient matrix to numtiply on the dist_coefs matrix 
        # to see how the Ks change the distortion of one normal image 
        coef_mat_numti = cp.array([1, 0, 0, 0, 0])
        dist_coefs = dist_coefs * coef_mat_numti

        # source: 1280 * 720
        # 172 * 100
        # 132 * 100
        rate_ = float(172/1280)


        # 大长方形
        camera_matrix = cp.array([[ 9.8214053893310370e+02*rate_    , 0.  , 172/2],
                                [   0.          ,9.8214053893310370e+02*rate_,  100/2],
                                [   0.            ,0.            ,1.        ]])
        # 大正方形
        # camera_matrix = np.array([[1.6773809437350340e+03, 0., 3.6150000000000000e+02],
        #                           [0.,  1.6773809437350340e+03, 3.6150000000000000e+02],
        #                           [0., 0., 1]])
        # 小正方形
        # camera_matrix = np.array([[1.9435494965485543e+02, 0., 5.5500000000000000e+01],
        #                           [0., 1.9435494965485543e+02, 5.5500000000000000e+01],
        #                           [0., 0., 1.]])
        # camera_matrix = np.transpose(camera_matrix)
        # camera_matrix = np.mat(camera_matrix).I
        # camera_matrix = camera_matrix * -10
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))        
        # dst = cv2.fisheye.undistortImage(img, camera_matrix, dist_coefs, None, newcameramtx)
        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        # print(newcameramtx)

        # x, y, w, h = [0,0,460, 340]
        # dst = dst[y:y+h, x:x+w]

        outfile = img_found + '_undistorted.png'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite('change_weight_1.500.png', dst)
 
    cv2.destroyAllWindows()
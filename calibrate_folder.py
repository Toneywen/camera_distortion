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
# local modules
from common import splitfn
# built-in modules
import os
 
if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob
 
    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1)
    # print(img_mask)
    if not img_mask:
        img_mask = "/home/wenxiangyu/project/camera_distortion/normal_2.jpg"  # default
    else:
        img_mask = img_mask[0]
    
    # print(img_mask)
    # img_mask = '/home/wenxiangyu/project/gesture-data/normal/gesture-left-cross/*.jpg'
 
    img_names = glob(img_mask)
    print('img_names',img_names)
    debug_dir = args.get('--debug')
    if not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))
 
    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    
    # print('pattern_points', pattern_points)

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
   
    for fn in img_names:
        print('processing %s... ' % fn, end='')
        # here the images we read in is RGB 
        img = cv2.imread(fn)
        # img = cv2.resize(img, (463, 344))
        # cv2.imshow('fdd', img)
        # cv2.waitKey(0)
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
            # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vis = img
            # cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            # ./output/ /home/wenxiangyu/test_gray2rgb 00002 .jpg
            # print('debug_dir, path, name, ext:', debug_dir, path, name, ext)
            # outfile = './output/00001_chess.png'
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

    camera_matrix = np.eye(3)
    # print('camera_matrix', camera_matrix)
    dist_coefs = np.zeros(4)
    calib_flags=cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
   
    # rms, camera_matrix, dist_coefs, rvecs, tvecs=cv2.fisheye.calibrate(obj_points, img_points, (w, h), camera_matrix, dist_coefs, None, None)
   
    # print("nRMS:", rms)
    # print("camera matrix:n", camera_matrix)
    # print("distortion coefficients: ", dist_coefs.ravel())
 
    # undistort the image with the calibration
    # print('')
    for img_found in img_names_undistort:
        # read image in rgb model
        img = cv2.imread(img_found)
        # cv2.imshow('fdd', img)
        # cv2.waitKey(0)

        h, w = img.shape[:2]
        # print(h, w)

        # dist_coefs = np.array([ 6*3.7816712271668235e-01, 6*9.7194428295751345e-01, 0, 0, -6*2.9169719384592412e+00])
        # the source data of camera matrix
        # dist_coefs = np.array([ -3.7816712271668235e-01, -9.7194428295751345e-01, 0, 0, 2.9169719384592412e+00])
        # the positive data of camera matrix, to distort the normal image 
        dist_coefs = np.array([ 3.7816712271668235e-01, 9.7194428295751345e-01, 0, 0, -2.9169719384592412e+00])
        
        # the coefficient matrix to numtiply on the dist_coefs matrix 
        # to see how the Ks change the distortion of one normal image 
        coef_mat_numti = np.array([1.5, 0, 1, 1, 0])
        dist_coefs = dist_coefs * coef_mat_numti

        # dist_coefs = np.array([ 0, 0, 0, 0, 0])
        # print()
        camera_matrix = np.array([[ 9.8214053893310370e+02    ,0.  ,6.3950000000000000e+02],
                                [   0.          ,9.8214053893310370e+02,  3.5950000000000000e+02],
                                [   0.            ,0.            ,1.        ]])
        # dist_coefs = 
        # camera_matrix = np.transpose(camera_matrix)
        # camera_matrix = np.mat(camera_matrix).I
        # camera_matrix = camera_matrix * -10
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))        
        # dst = cv2.fisheye.undistortImage(img, camera_matrix, dist_coefs, None, newcameramtx)
        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        # print(newcameramtx)

        # x, y, w, h = [0,0,460, 340]
        # dst = dst[y:y+h, x:x+w]

        # outfile = img_found + '_undistorted.png'
        outfile = img_found.split('/')[-1].split('_')[0] + '.jpg'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite('/home/wenxiangyu/test_file_out/'+outfile, dst)
 
    cv2.destroyAllWindows()
#include "opencv2/opencv.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;
 
int main()
{
    VideoCapture inputVideo(0);
    if (!inputVideo.isOpened())
    {
        cout << "Could not open the input video: " << endl;
        return -1;
    }
    Mat frame;
    Mat frameCalibration;
 
    inputVideo >> frame;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 4.450537506243416e+02;
    cameraMatrix.at<double>(0, 1) = 0.192095145445498;
    cameraMatrix.at<double>(0, 2) = 3.271489590204837e+02;
    cameraMatrix.at<double>(1, 1) = 4.473690628394497e+02;
    cameraMatrix.at<double>(1, 2) = 2.442734958206504e+02;
 
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -0.320311439187776;
    distCoeffs.at<double>(1, 0) = 0.117708464407889;
    distCoeffs.at<double>(2, 0) = -0.00548954846049678;
    distCoeffs.at<double>(3, 0) = 0.00141925006352090;
    distCoeffs.at<double>(4, 0) = 0;
 
    Mat view, rview, map1, map2;
    Size imageSize;
    imageSize = frame.size();
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
        getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
        imageSize, CV_16SC2, map1, map2);
 
 
    while (1) //Show the image captured in the window and repeat
    {
        inputVideo >> frame;              // read
        if (frame.empty()) break;         // check if at end
        remap(frame, frameCalibration, map1, map2, INTER_LINEAR);
        imshow("Origianl", frame);
        imshow("Calibration", frameCalibration);
        char key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')break;
    }
    return 0;
}
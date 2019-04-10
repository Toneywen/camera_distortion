# camera_distortion  
this repo is ready for making distortion  

### 该工程主要是为了完成相机广角畸变的逆变换，也就是提供了桶形畸变的方法   
### 该工程主要分为以下几个部分：
  
---> calibreate.py 是主文件，该文件能够使用python语言完成基本的桶形畸变操作  
.  
---> try001 文件夹是用以生成畸变参数以及相机内参的脚本，语言C++   
.  
---> try003xxx 文件夹是用以测试不同的k1、k2、k3对畸变的影响程度  
.  
---> try004 是使用python对c++语言的移植，能够使用python语言完成广角畸变的底层数学实现，  
     其中distortion.py完成了基本的重构，   
     而torch_distortion.py则使用pytorch的框架，使之能够使用gpu计算，减少对cpu的占用率

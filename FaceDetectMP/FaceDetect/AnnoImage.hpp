//
//  AnnoImage.hpp
//
//
/******************************************************************************************
本模块的作用在于，将计算的结果以可视化的方式打印在输入影像的拷贝上，便于查看算法的效果。
 
Author: Fu Xiaoqiang
Date:   2022/9/11
******************************************************************************************/

#ifndef ANNO_IMAGE_HPP
#define ANNO_IMAGE_HPP

#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#include "Common.hpp"


/******************************************************************************************
本函数的功能是，draw the Face Box and Six Keypoints on the annotation image.
*******************************************************************************************/
void AnnoFaceBoxKPs(Mat& annoImage, FaceInfo_Int& faceInfo);

//-----------------------------------------------------------------------------------------


#endif /* end of ANNO_IMAGE_HPP */

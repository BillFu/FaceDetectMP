//
//  Utils.hpp
//
//
/*
本模块提供一些简单的辅助性函数。
 
Author: Fu Xiaoqiang
Date:   2022/9/11
*/

#ifndef UTILS_HPP
#define UTILS_HPP

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**********************************************************************************************
将缩小版(大小为H*W)的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
同时，在“喂”之前，对像素值Normalization，使之变为Float，取值范围为[0.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
***********************************************************************************************/
void FeedInputWithNormalizedImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C);

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
将缩小版(大小为H*W)的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
同时，在“喂”之前，对像素值Quantization，使之变为Float，取值范围为[-1.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
***********************************************************************************************/
void FeedInputWithQuantizedImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C);

//-------------------------------------------------------------------------------------------

// return a string that present a float with 2 decimal digits.
// for example, return "3.14" for 3.1415927
string convertFloatToStr2DeciDigits(float value);

#endif /* end of UTILS_HPP */

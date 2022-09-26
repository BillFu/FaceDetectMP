//
//  Utils.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "Utils.hpp"
#include <array>
#include <limits>


// return a string that present a float with 2 decimal digits.
// for example, return "3.14" for 3.1415927
string convertFloatToStr2DeciDigits(float value)
{
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    std::string out_str = stream.str();
    return out_str;
}

//-------------------------------------------------------------------------------------------

/******************************************************************************************
将192*192的缩小版的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
这个函数为Face Detect和Face Mesh推理时所共享。
https://github.com/bferrarini/FloppyNet_TRO/blob/master/FloppyNet_TRO/TRO_pretrained/RPI4/src/lce_cnn.cc
see the function: ProcessInputWithFloatModel()
对像素值Normalization，使之变为Float，取值范围为[0.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
 *******************************************************************************************/
void FeedInputWithNormalizedImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C)
{
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (C == 3)
            {
                // here is TRICK: BGR ==> RGB, be careful with the channel indices coupling
                // happened on the two sides of assignment operator.
                // another thing to be noted: make the value of each channel of the pixels
                // in [0.0 1.0], dividing by 255.0
                int yWC = y * W * C;
                int xC = x * C;
                int yWC_xC = yWC + xC;
                *(netInputBuffer + yWC_xC + 0) = *(imgDataPtr + yWC_xC + 2) / 255.0f;
                *(netInputBuffer + yWC_xC + 1) = *(imgDataPtr + yWC_xC + 1) / 255.0f;
                *(netInputBuffer + yWC_xC + 2) = *(imgDataPtr + yWC_xC + 0) / 255.0f;
            }
            else
            {
                // only one channel, gray scale image
                {
                    *(netInputBuffer + y * W + x ) = *(imgDataPtr + y * W + x) / 255.0f;
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------------
// convert a uint8 to a float in the range [-1.0 1.0]
float Quantize(uint8_t gray_value)
{
    float result = (gray_value - 128.0f) / 128.0f;
    return result;
}

/**********************************************************************************************
将缩小版(大小为H*W)的影像“喂”给TF Lite网络的输入端，图像采用BGR通道次序。
同时，在“喂”之前，对像素值Quantization，使之变为Float，取值范围为[-1.0 1.0]。
imgDataPtr已经是缩小版的输入影像了。
***********************************************************************************************/
void FeedInputWithQuantizedImage(uint8_t* imgDataPtr, float* netInputBuffer, int H, int W, int C)
{
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (C == 3)
            {
                // here is TRICK: BGR ==> RGB, be careful with the channel indices coupling
                // happened on the two sides of assignment operator.
                // another thing to be noted: make the value of each channel of the pixels
                // in [0.0 1.0], dividing by 255.0
                int yWC = y * W * C;
                int xC = x * C;
                int yWC_xC = yWC + xC;
                *(netInputBuffer + yWC_xC + 0) = Quantize(imgDataPtr[yWC_xC + 2]);
                *(netInputBuffer + yWC_xC + 1) = Quantize(imgDataPtr[yWC_xC + 1]);
                *(netInputBuffer + yWC_xC + 2) = Quantize(imgDataPtr[yWC_xC + 0]);
            }
            else
            {
                // only one channel, gray scale image
                {
                    *(netInputBuffer + y * W + x ) = Quantize(imgDataPtr[y * W + x]);
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
 采用Padding的方式，将原始图像扩充为一个正方形。
 正方形的边长等于Max(srcWidth, srcHeight)，原始像素矩阵在正方形中居中。
 填充的像素取黑色。
***********************************************************************************************/
void MakeSquareImage(const Mat& srcImg, Mat& squareImg)
{
    int srcW = srcImg.cols;
    int srcH = srcImg.rows;
    
    if(srcW >= srcH) // this case will not happened in our task
        squareImg = srcImg.clone();
    else // srcW < srcH
    {
        Scalar blackColor(0, 0, 0);
        
        int padSideW = (srcH - srcW)/ 2;
        int top = 0;
        int bottom = 0;
        copyMakeBorder( srcImg, squareImg, top, bottom, padSideW, padSideW,
                       BORDER_CONSTANT, blackColor);
    }
}

//-------------------------------------------------------------------------------------------

/**********************************************************************************************
采用Padding的方式，将原始图像扩充为一个正方形。
上下左右都扩展。
先扩充上下，newH = srcH * alpha, alpha取[0.2 0.4]
依据newH，再扩展左右，使之成为一个正方形
正方形的边长等于newH，原始像素矩阵在正方形中居中。
填充的像素取黑色。
***********************************************************************************************/
void MakeSquareImageV2(const Mat& srcImg, float deltaHRatio, Mat& squareImg)
{
    int srcW = srcImg.cols;
    int srcH = srcImg.rows;
    
    //assert(srcW % 2 == 0); // must be a even number
    //assert(srcH % 2 == 0); // must be a even number

    if(srcW >= srcH) // this case will not happened in our task
        squareImg = srcImg.clone();
    else // srcW < srcH
    {
        Scalar blackColor(0, 0, 0);
        
        int newH = (int)(srcH*(1 + deltaHRatio));
        if(newH % 4 != 0) // should be divided by 4 completely
            newH += 4 - (newH % 4);
        
        int padVertTop = (newH - srcH) / 2;  // padding at top and bottom
        int padSideLeft = (newH - srcW)/ 2;
        int padVertBot = newH - padVertTop - srcH;
        int padSideRight = newH - padSideLeft - srcW;
        
        copyMakeBorder( srcImg, squareImg,
                       padVertTop, padVertBot, //
                       padSideLeft, padSideRight,
                       BORDER_CONSTANT, blackColor);
    }
}
//-------------------------------------------------------------------------------------------

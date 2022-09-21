//
//  AnnoImage.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/11

********************************************************************************/

#include "AnnoImage.hpp"
#include "Utils.hpp"


/******************************************************************************************
本函数的功能是，draw the Face Box and Six Keypoints on the annotation image.
*******************************************************************************************/
void AnnoFaceBoxKPs(Mat& annoImage, FaceInfo_Int& faceInfo)
{
    cv::Scalar colorBox(255, 0, 0); // (B, G, R)
    for(int i = 0; i < 468; i++)
    {
        int thickness = 2;
          
        // Drawing the Rectangle
        Point2i p1 = faceInfo.getP1();
        Point2i p2 = faceInfo.getP2();

        rectangle(annoImage, p1, p2,
                  colorBox, thickness, LINE_8);
    }
    
    for(int i=0; i<NUM_KP_IN_FACE; i++)
    {
        int x = faceInfo.keyPts[i].x;
        int y = faceInfo.keyPts[i].y;
        
        cv::Point center(x, y);
        cv::circle(annoImage, center, 5, colorBox, cv::FILLED);
    }
}

//-----------------------------------------------------------------------------------------


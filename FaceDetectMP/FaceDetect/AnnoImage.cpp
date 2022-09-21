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
    
    /*
    // 对应Dlib上点的序号为18, 22, 23, 27, 37, 40, 43, 46, 32, 36, 49, 55, 58, 9
    int face_2d_pts_indices[] = {46, 55, 285, 276, 33, 173,
        398, 263, 48, 278, 61, 291, 17, 199};  // indics in face lms in mediapipe.
        
    cv::Scalar yellow(0, 255, 255); // (B, G, R)

    for(int i=0; i<14; i++)
    {
        int lm_index = face_2d_pts_indices[i];
        int x = faceInfo.lm_2d[lm_index][0];
        int y = faceInfo.lm_2d[lm_index][1];
        
        cv::Point center(x, y);
        cv::circle(annoImage, center, 5, yellow, cv::FILLED);
    }
    */
}

//-----------------------------------------------------------------------------------------


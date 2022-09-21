//
//  Common.hpp
//
//
/*
本模块提供一些基础性的公共定义，供各模块使用。
 
Author: Fu Xiaoqiang
Date:   2022/9/21
*/

#ifndef COMMON_HPP
#define COMMON_HPP

using namespace std;
using namespace cv;

#define  NUM_KP_IN_FACE    6    // the number of key points in one face

struct FaceIndexScore
{
    int index;  // take a value in [0 896]
    float score;
    
    FaceIndexScore(int index0, float score0)
    {
        index = index0;
        score = score0;
    }
};

struct FaceInfo_Float
{
    // x1, y1, x2, y2与后面的box存在信息冗余，但为NMS算法的便利，故意这么做。
    float x1;
    float y1;
    float x2;
    float y2;
    
    Rect_<float> box;
    Point2f keyPts[NUM_KP_IN_FACE];
    float score;
    
    void setBox(const Point2f& pt1, const Point2f& pt2)
    {
        box = Rect_<float>(pt1, pt2);
        x1 = pt1.x;
        y1 = pt1.y;
        x2 = pt2.x;
        y2 = pt2.y;
    }
    
    Point2f getP1()
    {
        return Point2f(x1, y1);
    }
    
    Point2f getP2()
    {
        return Point2f(x2, y2);
    }
};

struct FaceInfo_Int
{
    // x1, y1, x2, y2与后面的box存在信息冗余，但为NMS算法的便利，故意这么做。
    int x1;
    int y1;
    int x2;
    int y2;
    
    Rect_<int> box;
    Point2i keyPts[NUM_KP_IN_FACE];
    float score;
    
    void setBox(const Point2i& pt1, const Point2i& pt2)
    {
        box = Rect_<int>(pt1, pt2);
        x1 = pt1.x;
        y1 = pt1.y;
        x2 = pt2.x;
        y2 = pt2.y;
    }
    
    void setBox(int xa, int ya, int xb, int yb)
    {
        x1 = xa;
        y1 = ya;
        x2 = xb;
        y2 = yb;
        
        box = Rect_<int>(Point2i(x1, y1), Point2i(x2, y2));
    }
    
    Point2i getP1()
    {
        return Point2i(x1, y1);
    }
    
    Point2i getP2()
    {
        return Point2i(x2, y2);
    }
};

#endif /* end of COMMON_HPP */

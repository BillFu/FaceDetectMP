//
//
/************************************************************************************************************
本模块的功能是，利用tensorflow lite C++ API来驱动MediaPipe中内含的Blaze Face Detection Model.
本模块由python代码改写而来，原始代码出处如下：
https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference/blob/main/BlazeFaceDetection/blazeFaceDetector.py

本模块采用Front相机模型，效果可能比Back相机模型稍差。
采用Back相机模型的推理引擎参加另外的模块。
 
Author: Fu Xiaoqiang
Date:   2022/9/21
*************************************************************************************************************/

#ifndef BLAZE_FACE_DETECTOR_H
#define BLAZE_FACE_DETECTOR_H

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "Common.hpp"
#include "BlazeFaceUtil.h"


using namespace std;
using namespace tflite;
using namespace cv;

#define  FACE_DETECT_FRONT_MODEL_SIZE  229032


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

class BlazeFaceDetector
{
private:
    unique_ptr<FlatBufferModel> mFrontModel;
    unique_ptr<Interpreter> mInterpreter;
    
    vector<Anchor> mAnchors;
    
    int mNetInputHeight;  //对应于python代码中的self.inputHeight
    int mNetInputWidth;   //对应于python代码中的self.inputWidth
    int mNetChannels;
    
    float mScoreThreshold;
    float mIouThreshold;  // ???
    
    float mSigScoreTh;  // Sigmoid Score Threshold

    int mImgHeight;
    int mImgWidth;
    int mImgChannels;
    
public:
    static bool isFrontModelBufFilled;
    static char frontModelBuffer[FACE_DETECT_FRONT_MODEL_SIZE];
    
    // 这个函数在程序初始化时要调用一次，并确保返回true之后，才能往下进行
    static bool loadFrontModelFile(const string& modelFileName);
    
    BlazeFaceDetector(float scoreThreshold = 0.7, float iouThreshold = 0.3);
    ~BlazeFaceDetector();
    
    bool initFrontModel();

    bool DetectFaces(const Mat& srcImage, vector<FaceInfo_Int>& outFaceInfoSet);

    //void convFaceInfo2SrcImgCoordinate(const FaceInfo& oldInfo, FaceInfo_Int& newInfo);
    
private:
    void getModelInputDetails();
    
    void genFrontModelAnchors();
        
    void filterDetections(vector<FaceIndexScore>& indexScoreCds); // Cd stands for candidate
    
    void extractDetections(const vector<FaceIndexScore>& indexScoreCds,
                           vector<FaceInfo_Float>& faceInfoCds);
    
    ////NMS is the abbreviation for NonMaxSupression
    void NMS(vector<FaceInfo_Float>& inFaceSet,
            vector<FaceInfo_Float>& outFaceSet);
    
    // 将计算结果转换到原始输入图像的坐标空间中
    void convFaceInfo2SrcImgCoordinate(const FaceInfo_Float& oldInfo, FaceInfo_Int& newInfo);
    
    int convX2SrcImgCoordinate(float x);
    int convY2SrcImgCoordinate(float y);

};

#endif // BLAZE_FACE_DETECTOR_H

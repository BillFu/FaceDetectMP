//
// https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference/blob/main/BlazeFaceDetection/blazeFaceDetector.py
//

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

#include "BlazeFaceUtil.h"


using namespace std;
using namespace tflite;
using namespace cv;

#define  FACE_DETECT_FRONT_MODEL_SIZE  229032


enum { N_FACE_ATTB=5 }; // number of attributes of the following struct:

struct FaceInfo{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

class BlazeFaceDetector
{
private:
    unique_ptr<FlatBufferModel> mFrontModel;
    unique_ptr<Interpreter> mInterpreter;
    
    vector<Anchor> mAnchors;
    
    int mNetInputHeight;
    int mNetInputWidth;
    int mNetChannels;
    
    float mScoreThreshold;
    float mIouThreshold;
    
    float mSigScoreTh;  // Sigmoid Score Threshold

    int mImg_height;
    int mImg_width;
    int mImg_channels;
    
public:
    static bool isFrontModelBufFilled;
    static char frontModelBuffer[FACE_DETECT_FRONT_MODEL_SIZE];
    
    // 这个函数在程序初始化时要调用一次，并确保返回true之后，才能往下进行
    static bool loadFrontModelFile(const string& modelFileName);
    
    BlazeFaceDetector(float scoreThreshold = 0.7, float iouThreshold = 0.3);
    ~BlazeFaceDetector();
    
    bool initFrontModel();

    bool DetectFaces(const Mat& srcImage);

private:
    void getModelInputDetails();
    
    void genFrontModelAnchors();
    
    //Mat preInputForInference(const Mat& srcImage);
    
    void filterDetections(vector<int>& goodIndices,
                          vector<float>& goodScore);

};

#endif // BLAZE_FACE_DETECTOR_H

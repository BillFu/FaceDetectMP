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

using namespace std;
using namespace tflite;

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
    /*
    char* mModelBuffer = nullptr;
    long mModelSize;
    bool mModelQuantized = false;
    const int OUTPUT_WIDTH = 640;
    const int OUTPUT_HEIGHT = 480;
     */
    
    unique_ptr<FlatBufferModel> mFrontModel;
    unique_ptr<Interpreter> mInterpreter;
    /*
    TfLiteTensor* mInputTensor = nullptr;
    TfLiteTensor* mOutputHeatmap = nullptr;
    TfLiteTensor* mOutputScale = nullptr;
    TfLiteTensor* mOutputOffset = nullptr;

    int d_h;
    int d_w;
    float d_scale_h;
    float d_scale_w;
    float scale_w ;
    float scale_h ;
    int image_h;
    int image_w;
    */

public:
    static char frontModelBuffer[FACE_DETECT_FRONT_MODEL_SIZE];
    static bool loadFrontModelFile(const string& modelFileName);
    
    BlazeFaceDetector();  //char* buffer, long size, bool quantized=false);
    ~BlazeFaceDetector();
    
    bool initFrontModel();

    //void detect(cv::Mat img, std::vector<FaceInfo>& faces,
    //        float scoreThresh, float nmsThresh);

private:
    void getModelInputDetails();
    void getModelOutputDetails();
    
    void genFrontModelAnchors();

    /*
    void dynamic_scale(float in_w, float in_h);
    void postProcess(float* heatmap, float* scale, float* offset,
                     std::vector<FaceInfo>& faces,
                     float heatmapThreshold, float nmsThreshold);
    void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,
            float nmsThreshold);
    std::vector<int> filterHeatmap(float* heatmap, int h, int w, float thresh);
    void getBox(std::vector<FaceInfo>& faces);
     */
};

#endif // BLAZE_FACE_DETECTOR_H

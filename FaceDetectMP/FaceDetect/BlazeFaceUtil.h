//
// https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference/blob/main/BlazeFaceDetection/blazeFaceUtils.py
//

#ifndef BLAZE_FACE_UTIL_H
#define BLAZE_FACE_UTIL_H

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

class SsdAnchorsCalcOpts
{
public:
    int input_size_width;
    int input_size_height;
    
    float mMin_scale;
    float mMax_scale;
    
    float mAnchor_offset_x;
    float mAnchor_offset_y;
    
    int mNum_layers;
    
    vector<int> mFeature_map_width;
    vector<int> mFeature_map_height;
    
    int mFeature_map_width_size;
    int mFeature_map_height_size;
    
    vector<int> mStrides;
    int mStrides_size;
    
    vector<float> mAspect_ratios;
    int mAspect_ratios_size;

    bool mReduce_boxes_in_lowest_layer;
    
    float mInterpolated_scale_aspect_ratio;
    bool mFixed_anchor_size;
    
public:
    SsdAnchorsCalcOpts(int input_size_width, int input_size_height,
                       float min_scale, float max_scale,
                       int num_layers,
                       const vector<int>& feature_map_width,
                       const vector<int>& feature_map_height,
                       const vector<int>& strides,
                       const vector<float>& aspect_ratios,
                       float anchor_offset_x=0.5,
                       float anchor_offset_y=0.5,
                       bool reduce_boxes_in_lowest_layer=false,
                       float interpolated_scale_aspect_ratio=1.0,
                       bool fixed_anchor_size=false);
                    
    ~SsdAnchorsCalcOpts();
    
};

class Anchor
{
public:
    int x_center;
    int y_center;
    int w, h;
    
public:
    Anchor(int x_center0, int y_center0, int w0, int h0);
};

std::ostream& operator<<(ostream &strm, const Anchor &a);

void genAnchors(const SsdAnchorsCalcOpts& options, vector<Anchor>& anchors);

#endif // BLAZE_FACE_UTIL_H

// https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference/blob/main/BlazeFaceDetection/blazeFaceUtils.py

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#include "BlazeFaceUtil.h"

Anchor::Anchor(float x_center0, float y_center0, float w0, float h0)
{
    x_center = x_center0;
    y_center = y_center0;
    h = h0;
    w = w0;
}

std::ostream& operator<<(ostream &strm, const Anchor &a)
{
    return strm << "Anchor(x_center: " << a.x_center
        << ", y_center: " << a.y_center
        << ", w: " << a.w
        << ", h: " << a.h << ")";
}

SsdAnchorsCalcOpts::SsdAnchorsCalcOpts(int input_size_width, int input_size_height,
                                       float min_scale, float max_scale,
                                       int num_layers,
                                       const vector<int>& feature_map_width,
                                       const vector<int>& feature_map_height,
                                       const vector<int>& strides,
                                       const vector<float>& aspect_ratios,
                                       float anchor_offset_x,
                                       float anchor_offset_y,
                                       bool reduce_boxes_in_lowest_layer,
                                       float interpolated_scale_aspect_ratio,
                                       bool fixed_anchor_size)
{
    // Size of input images.
    this->input_size_width = input_size_width;
    this->input_size_height = input_size_height;
    
    // Min and max scales for generating anchor boxes on feature maps.
    mMin_scale = min_scale;
    mMax_scale = max_scale;
    
    // Number of output feature maps to generate the anchors on.
    mNum_layers = num_layers;
    /* Sizes of output feature maps to create anchors. Either feature_map size or
     stride should be provided.*/
    
    mFeature_map_width = feature_map_width;
    mFeature_map_height = feature_map_height;
    
    mFeature_map_width_size = (int)(feature_map_width.size());
    mFeature_map_height_size = (int)(feature_map_height.size());
    
    // Strides of each output feature maps.
    mStrides = strides;
    mStrides_size = (int)(mStrides.size());
    
    // List of different aspect ratio to generate anchors.
    mAspect_ratios = aspect_ratios;
    mAspect_ratios_size = (int)(mAspect_ratios.size());

    /* The offset for the center of anchors. The value is in the scale of stride.
     E.g. 0.5 meaning 0.5 * |current_stride| in pixels. */
    mAnchor_offset_x = anchor_offset_x;
    mAnchor_offset_y = anchor_offset_y;
    
    // A boolean to indicate whether the fixed 3 boxes per location is used in the lowest layer.
    mReduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer;
    
    /* An additional anchor is added with this aspect ratio and a scale
        interpolated between the scale for a layer and the scale for the next layer
        (1.0 for the last layer). This anchor is not included if this value is 0. */
    mInterpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio;
    
    /* Whether use fixed width and height (e.g. both 1.0f) for each anchor.
       This option can be used when the predicted anchor width and height are in  pixels. */
    mFixed_anchor_size = fixed_anchor_size;
}

SsdAnchorsCalcOpts::~SsdAnchorsCalcOpts()
{
    
}

void genAnchors(const SsdAnchorsCalcOpts& options, vector<Anchor>& anchors)
{
    // # Verify the options.
    if (options.mStrides_size != options.mNum_layers)
    {
        cout << "strides_size and num_layers must be equal!!!." << endl;
        return;
    }
    
    int layer_id = 0;
    while (layer_id < options.mStrides_size)
    {
        vector<float> anchor_height;
        vector<float> anchor_width;
        vector<float> aspect_ratios;
        vector<float> scales;
        
        //For same strides, we merge the anchors in the same order.
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < options.mStrides_size &&
               options.mStrides[last_same_stride_layer] == options.mStrides[layer_id])
        {
            float scale = options.mMin_scale + (options.mMax_scale - options.mMin_scale) * 1.0 * last_same_stride_layer / (options.mStrides_size - 1.0);
            
            if (last_same_stride_layer == 0 and options.mReduce_boxes_in_lowest_layer)
            {
                // For first layer, it can be specified to use predefined anchors.
                aspect_ratios.push_back(1.0);
                aspect_ratios.push_back(2.0);
                aspect_ratios.push_back(0.5);
                scales.push_back(0.1);
                scales.push_back(scale);
                scales.push_back(scale);
            }
            else
            {
                for(auto aspect_ratio:options.mAspect_ratios)
                {
                    aspect_ratios.push_back(aspect_ratio);
                    scales.push_back(scale);
                }
                
                if (options.mInterpolated_scale_aspect_ratio > 0.0)
                {
                    float scale_next;
                    if(last_same_stride_layer == options.mStrides_size - 1)
                        scale_next = 1.0;
                    else
                        scale_next = options.mMin_scale + (options.mMax_scale - options.mMin_scale) * (last_same_stride_layer+1) / (options.mStrides_size - 1.0);
                    
                    scales.push_back(sqrt(scale * scale_next));
                    aspect_ratios.push_back(options.mInterpolated_scale_aspect_ratio);
                }
            }
            
            last_same_stride_layer += 1;
        }
        
        for(int i=0; i<aspect_ratios.size(); i++)
        {
            float ratio_sqrts = sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }
         
        int feature_map_height = 0;
        int feature_map_width = 0;
        
         if (options.mFeature_map_height_size > 0)
         {
             feature_map_height = options.mFeature_map_height[layer_id];
             feature_map_width = options.mFeature_map_width[layer_id];
         }
         else
         {
             int stride = options.mStrides[layer_id];
             feature_map_height = ceil(1.0 * options.input_size_height / stride);
             feature_map_width = ceil(1.0 * options.input_size_width / stride);
         }
         
        for(int y=0; y<feature_map_height; y++)
            for(int x=0; x<feature_map_width; x++)
                for(int anchor_id=0; anchor_id < anchor_height.size(); anchor_id++)
                {
                    //# TODO: Support specifying anchor_offset_x, anchor_offset_y.
                    float x_center = (x + options.mAnchor_offset_x) * 1.0 / feature_map_width;
                    float y_center = (y + options.mAnchor_offset_y) * 1.0 / feature_map_height;
                    float w = 0;
                    float h = 0;
                    if (options.mFixed_anchor_size)  // !!!
                    {
                        w = 1.0;
                        h = 1.0;
                    }
                    else
                    {
                        w = anchor_width[anchor_id];
                        h = anchor_height[anchor_id];
                    }
                    Anchor new_anchor(x_center, y_center, w, h);
                    anchors.push_back(new_anchor);
                 }

        layer_id = last_same_stride_layer;
    }
}

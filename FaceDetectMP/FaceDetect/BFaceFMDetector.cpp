
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#include "BFaceBMDetector.h"
#include "BlazeFaceUtil.h"
#include "Utils.hpp"

bool BFaceBMDetector::isModelBufFilled = false;
char BFaceBMDetector::backModelBuffer[FACE_DETECT_BACK_MODEL_SIZE];

bool BFaceBMDetector::loadModelFile(const string& modelFileName)
{
    ifstream modelFile(modelFileName.c_str(), ios::in | ios::binary);
    
    modelFile.read(backModelBuffer, FACE_DETECT_BACK_MODEL_SIZE);
    if(!modelFile)
    {
        cout << "Failed to load the model file: " << modelFileName << endl;
        return false;
    }
    else
    {
        cout << "Model file has been Successfully loaded!" << endl;
        isModelBufFilled = true;
        return true;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

BFaceBMDetector::BFaceBMDetector(float scoreThreshold, float iouThreshold)
{
    //self.type = type
    mScoreThreshold = scoreThreshold;
    mIouThreshold = iouThreshold;
    mSigScoreTh = log(mScoreThreshold / (1.0 - mScoreThreshold));

    // Initialize model based on model type
    initBackModel();

    // # Generate anchors for model
    genModelAnchors();
}

BFaceBMDetector::~BFaceBMDetector()
{
}

bool BFaceBMDetector::initBackModel()
{
    if(!isModelBufFilled)
    {
        cout << "To invoke BFaceBMDetector::loadBackModelFile() first!" << endl;
        return false;
    }
        
    mModel = FlatBufferModel::BuildFromBuffer(backModelBuffer, FACE_DETECT_BACK_MODEL_SIZE);
    if(mModel == nullptr)
        return false;

        // Build the interpreter with the InterpreterBuilder.
    ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*mModel, resolver);
    builder(&mInterpreter);
    if(mInterpreter == nullptr)
        return false;

    if(mInterpreter->AllocateTensors() != kTfLiteOk)
        return false;

    //# Get model info
    getModelInputDetails();
        
    return true;
}

void BFaceBMDetector::getModelInputDetails()
{
    // Get Input Tensor Dimensions
    int inTensorIndex = mInterpreter->inputs()[0];
    mNetInputHeight = mInterpreter->tensor(inTensorIndex)->dims->data[1];
    mNetInputWidth = mInterpreter->tensor(inTensorIndex)->dims->data[2];
    mNetChannels = mInterpreter->tensor(inTensorIndex)->dims->data[3];
}

void BFaceBMDetector::genModelAnchors()
{
    vector<int> feature_map_width;
    vector<int> feature_map_height;
    vector<int> strides = {16, 32, 32, 32};
    vector<float> aspect_ratios = {1.0};
    //vector<float> aspect_ratios = {0.75};

    // Options to generate anchors for SSD object detection models.
     SsdAnchorsCalcOpts options(256, 256, 0.15625, 0.75, 4,
                                feature_map_width, feature_map_height,
                                strides, aspect_ratios,
                                0.5, 0.5, //anchor_offset_x, anchor_offset_y
                                false, 1.0, true);
    
    genAnchors(options, mAnchors);
    
    for(Anchor anchor : mAnchors)
    {
        cout << anchor << endl;
    }
}

bool BFaceBMDetector::DetectFaces(const Mat& srcImage,
                                    vector<FaceInfo_Int>& outFaceInfoSet)
{
    mImgHeight = srcImage.rows;
    mImgWidth = srcImage.cols;
    mImgChannels = srcImage.channels();
    
    int inTensorIndex = mInterpreter->inputs()[0];
    
    Mat resized_image;
    // Not need to perform the convertion from BGR to RGB by the noticeable statements,
    // later it would be done in one trick way.
    resize(srcImage, resized_image, Size(mNetInputWidth, mNetInputHeight), INTER_NEAREST);
    
    float* inTensorBuf = mInterpreter->typed_input_tensor<float>(inTensorIndex);
    uint8_t* inImgMem = resized_image.ptr<uint8_t>(0);
    FeedInputWithQuantizedImage(inImgMem, inTensorBuf, mNetInputHeight, mNetInputWidth, mNetChannels);
    cout << "FeedInputWithQuantizedImage() is executed successfully!" << endl;
    
    // Inference
    if (mInterpreter->Invoke() != kTfLiteOk)
    {
        cout << "!!! Failed to execute: mInterpreter->Invoke()." << endl;
        return false;
    }

    vector<FaceIndexScore> indexScoreCds;
    // Filter scores based on the detection scores
    filterDetections(indexScoreCds);
    
    vector<FaceInfo_Float> faceInfoCds;
    extractDetections(indexScoreCds, faceInfoCds);
    
    vector<FaceInfo_Float> outFaceInfoSet_LC;  // local coordinate
    NMS(faceInfoCds, outFaceInfoSet_LC);
    
    for(FaceInfo_Float oldInfo: outFaceInfoSet_LC)
    {
        FaceInfo_Int newInfo;
        convFaceInfo2SrcImgCoordinate(oldInfo, newInfo);
        outFaceInfoSet.push_back(newInfo);
    }
    return true;
}

// Cds for Candidates
void BFaceBMDetector::filterDetections(vector<FaceIndexScore>& indexScoreCds)
{
    //int outConfID = mInterpreter->outputs()[1];  // confidence
    //cout << "output confidence ID: " << output_conf_ID << endl;
    float* confPtr = mInterpreter->typed_output_tensor<float>(1);
    
    cout << "mSigScoreTh: " << mSigScoreTh << endl;
    for(int i=0; i<896; i++)
    {
        //cout << i << ", " << confPtr[i] << endl;
        if(confPtr[i] > mSigScoreTh)
        {
            float score = 1.0 /(1.0 + exp(-confPtr[i]));
            FaceIndexScore indexScore(i, score);
            indexScoreCds.push_back(indexScore);
        }
    }
}

void BFaceBMDetector::extractDetections(
        const vector<FaceIndexScore>& indexScoreCds,
        vector<FaceInfo_Float>& faceInfoCds)
{
    // bounding box and six key points
    float* out0Ptr = mInterpreter->typed_output_tensor<float>(0);
    //float boxKeyPt[][16]

    for(FaceIndexScore indexScore: indexScoreCds)
    {
        FaceInfo_Float faceInfo;
        faceInfo.score = indexScore.score;
        
        const Anchor& anchor = mAnchors[indexScore.index];

        float* pBoxKeyPts = out0Ptr + indexScore.index * 16;
        float sx = pBoxKeyPts[0];  // scaled x in [0.0 1.0]
        float sy = pBoxKeyPts[1];  // scaled y in [0.0 1.0]
        float w = pBoxKeyPts[2];
        float h = pBoxKeyPts[3];
        
        float cx = sx + anchor.x_center * mNetInputWidth;
        float cy = sy + anchor.y_center * mNetInputHeight;

        cx /= mNetInputWidth;
        cy /= mNetInputHeight;
        w /= mNetInputWidth;
        h /= mNetInputHeight;
        
        Point2f pt1(cx - w * 0.5, cy - h * 0.5);
        Point2f pt2(cx + w * 0.5, cy + h * 0.5);

        faceInfo.setBox(pt1, pt2);
        
        float* pKeyPts = pBoxKeyPts + 4;
        for(int ptIndex = 0; ptIndex < NUM_KP_IN_FACE; ptIndex++)
        {
            float* pCurKeyPt = pKeyPts + 2*ptIndex;
            float lx = pCurKeyPt[0];
            float ly = pCurKeyPt[1];
            
            lx += anchor.x_center * mNetInputWidth;
            ly += anchor.y_center * mNetInputHeight;
            
            lx /= mNetInputWidth;
            ly /= mNetInputHeight;
            
            faceInfo.keyPts[ptIndex] = Point2f(lx, ly);
        }
        
        faceInfoCds.push_back(faceInfo);
    }
}

// https://github.com/cuongvng/Face-Detection-TFLite-JNI-Android/blob/master/app/src/main/cpp/face-detection.cpp

void BFaceBMDetector::NMS(vector<FaceInfo_Float>& inFaceSet,
                            vector<FaceInfo_Float>& outFaceSet)
{
    // sort the elements in inFaceSet by score in Descending order
    std::sort(inFaceSet.begin(), inFaceSet.end(),
        [](const FaceInfo_Float& a, const FaceInfo_Float& b){return a.score > b.score;});

    int box_num = (int)(inFaceSet.size());

    std::vector<int> merged(box_num, 0);
    for (int i = 0; i < box_num; i++)
    {
        if (merged[i])
            continue;

        outFaceSet.push_back(inFaceSet[i]);

        float h0 = inFaceSet[i].y2 - inFaceSet[i].y1 + 1;
        float w0 = inFaceSet[i].x2 - inFaceSet[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++)
        {
            if (merged[j])
                continue;

            float inner_x0 = inFaceSet[i].x1 > inFaceSet[j].x1 ? inFaceSet[i].x1 : inFaceSet[j].x1;
            float inner_y0 = inFaceSet[i].y1 > inFaceSet[j].y1 ? inFaceSet[i].y1 : inFaceSet[j].y1;

            float inner_x1 = inFaceSet[i].x2 < inFaceSet[j].x2 ? inFaceSet[i].x2 : inFaceSet[j].x2;
            float inner_y1 = inFaceSet[i].y2 < inFaceSet[j].y2 ? inFaceSet[i].y2 : inFaceSet[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = inFaceSet[j].y2 - inFaceSet[j].y1 + 1;
            float w1 = inFaceSet[j].x2 - inFaceSet[j].x1 + 1;
            float area1 = h1 * w1;

            float iou = inner_area / (area0 + area1 - inner_area);
            if (iou > mIouThreshold)
                merged[j] = 1;
        }

    }
}

int BFaceBMDetector::convX2SrcImgCoordinate(float x0)
{
    int x1 = (int)(x0 * mImgWidth);
    return x1;
}

int BFaceBMDetector::convY2SrcImgCoordinate(float y0)
{
    int y1 = (int)(y0 * mImgHeight);
    return y1;
}

// 将计算结果转换到原始输入图像的坐标空间中
void BFaceBMDetector::convFaceInfo2SrcImgCoordinate(
        const FaceInfo_Float& oldInfo, FaceInfo_Int& newInfo)
{
    int x1 = convX2SrcImgCoordinate(oldInfo.x1);
    int y1 = convY2SrcImgCoordinate(oldInfo.y1);
    
    int x2 = convX2SrcImgCoordinate(oldInfo.x2);
    int y2 = convY2SrcImgCoordinate(oldInfo.y2);
    
    newInfo.setBox(x1, y1, x2, y2);
    newInfo.score = oldInfo.score;
    
    for(int i = 0; i < NUM_KP_IN_FACE; i++)
    {
        Point2f p = oldInfo.keyPts[i];
        int x = convX2SrcImgCoordinate(p.x);
        int y = convY2SrcImgCoordinate(p.y);
        newInfo.keyPts[i] = Point2i(x, y);
    }
    
}

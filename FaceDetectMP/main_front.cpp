//
//  main.cpp
//  FaceDetectMP
//
//  Created by meicet on 2022/9/20.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "nlohmann/json.hpp"

#include <opencv2/opencv.hpp>
#include "FaceDetect/BlazeFaceDetector.h"
#include "FaceDetect/AnnoImage.hpp"

using namespace std;
using namespace cv;

using json = nlohmann::json;

int main(int argc, const char * argv[])
{
    if (argc != 2)
    {
        cout << "{target} config_file" << endl;
        return 0;
    }
    
    json config_json;            // 创建 json 对象
    ifstream jfile(argv[1]);
    jfile >> config_json;        // 以文件流形式读取 json 文件
        
    string faceDetectModelFile = config_json.at("FaceDetectModelFile");
    string srcImgFile = config_json.at("SourceImage");
    string annoImgFile = config_json.at("AnnoImage");
    
    cout << "faceDetectModelFile: " << faceDetectModelFile << endl;
    cout << "srcImgFile: " << srcImgFile << endl;
    cout << "annoImgFile: " << annoImgFile << endl;
    
    // Load Input Image
    Mat srcImage = cv::imread(srcImgFile.c_str());
    if(srcImage.empty())
    {
        cout << "Failed to load input iamge: " << srcImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << srcImgFile << endl;
    
    bool isOK = BlazeFaceDetector::loadFrontModelFile(faceDetectModelFile);
    if(!isOK)
    {
        cout << "Failed to load model file: " << faceDetectModelFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load model file: " << faceDetectModelFile << endl;

    float scoreThreshold = 0.75; //;
    float iouThreshold = 0.3;
    BlazeFaceDetector detector(scoreThreshold, iouThreshold);

    vector<FaceInfo_Int> outFaceInfoSet;
    isOK = detector.DetectFaces(srcImage, outFaceInfoSet);
    if(!isOK)
    {
        cout << "Failed to invoke BlazeFaceDetector::DetectFaces()."  << endl;
        return 0;
    }
    else
        cout << "Succeeded to invoke BlazeFaceDetector::DetectFaces()." << endl;
    
    //FaceInfo_Int faceInfo_GC = outFaceInfoSet[0]; // global coordinate

    for(FaceInfo_Int faceInfo_GC: outFaceInfoSet)
    {
        AnnoFaceBoxKPs(srcImage, faceInfo_GC);
    }
    
    imwrite(annoImgFile, srcImage);

    return 0;
}

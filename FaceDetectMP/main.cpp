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
    
    
    return 0;
}

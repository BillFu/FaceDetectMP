
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#include "BlazeFaceDetector.h"
#include "BlazeFaceUtil.h"

bool BlazeFaceDetector::isFrontModelBufFilled = false;
char BlazeFaceDetector::frontModelBuffer[FACE_DETECT_FRONT_MODEL_SIZE];

bool BlazeFaceDetector::loadFrontModelFile(const string& modelFileName)
{
    ifstream modelFile(modelFileName.c_str(), ios::in | ios::binary);
    
    modelFile.read(frontModelBuffer, FACE_DETECT_FRONT_MODEL_SIZE);
    if(!modelFile)
    {
        cout << "Failed to load the model file: " << modelFileName << endl;
        return false;
    }
    else
    {
        cout << "Model file has been Successfully loaded!" << endl;
        isFrontModelBufFilled = true;
        return true;
    }
}

BlazeFaceDetector::BlazeFaceDetector(float scoreThreshold, float iouThreshold)
{
    //self.type = type
    mScoreThreshold = scoreThreshold;
    mIouThreshold = iouThreshold;
    mSigmoidScoreThreshold = log(mScoreThreshold/(1.0-mScoreThreshold));

    // Initialize model based on model type
    initFrontModel();

    // # Generate anchors for model
    genFrontModelAnchors();
}

bool BlazeFaceDetector::initFrontModel()
{
    if(!isFrontModelBufFilled)
    {
        cout << "To invoke BlazeFaceDetector::loadFrontModelFile() first!" << endl;
        return false;
    }
        
    mFrontModel = FlatBufferModel::BuildFromBuffer(frontModelBuffer, FACE_DETECT_FRONT_MODEL_SIZE);
    assert(mFrontModel != nullptr);
    if(mFrontModel == nullptr)
        return false;

        // Build the interpreter with the InterpreterBuilder.
    ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*mFrontModel, resolver);
    builder(&mInterpreter);
    if(mInterpreter == nullptr)
        return false;

    if(mInterpreter->AllocateTensors() != kTfLiteOk)
        return false;

    //# Get model info
    getModelInputDetails();
    getModelOutputDetails();
        
    return true;
}

void BlazeFaceDetector::getModelInputDetails()
{
    // Get Input Tensor Dimensions
    int inTensorIndex = mInterpreter->inputs()[0];
    mNetInputHeight = mInterpreter->tensor(inTensorIndex)->dims->data[1];
    mNetInputWidth = mInterpreter->tensor(inTensorIndex)->dims->data[2];
    mNetChannels = mInterpreter->tensor(inTensorIndex)->dims->data[3];
}

void BlazeFaceDetector::getModelOutputDetails()
{
    //mOutputDetails = mInterpreter.get_output_details();
}

void BlazeFaceDetector::genFrontModelAnchors()
{
    vector<int> feature_map_width;
    vector<int> feature_map_height;
    vector<int> strides = {8, 16, 16, 16};
    vector<float> aspect_ratios = {1.0};
    
    // Options to generate anchors for SSD object detection models.
     SsdAnchorsCalcOpts options(128, 128, 0.1484375, 0.75, 4,
                                feature_map_width, feature_map_height,
                                strides, aspect_ratios,
                                0.5, 0.5,
                                false, 1.0, true);
    genAnchors(options, mAnchors);
}

/*

void FaceDetector::detect(cv::Mat img, std::vector<FaceInfo>& faces,
		float heatmapThreshold, float nmsThreshold){
	image_h = img.rows;
	image_w = img.cols;
	scale_w = (float)image_w / (float)d_w;
	scale_h = (float)image_h / (float)d_h;

    // Read image into `mInputTensor`
	cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1.0, cv::Size(d_w, d_h),
											cv::Scalar(0, 0, 0),
											true, CV_32F);

	memcpy(tflite::GetTensorData<float>(mInputTensor), inputBlob.data,
		   sizeof(float) * inputBlob.size[1] * inputBlob.size[2] * inputBlob.size[3]);
	assert(mInputTensor->type == kTfLiteFloat32);

    // Inference
	if (mInterpreter->Invoke() != kTfLiteOk){
		return;
	}

	postProcess(
		tflite::GetTensorData<float>(mOutputHeatmap),
		tflite::GetTensorData<float>(mOutputScale),
		tflite::GetTensorData<float>(mOutputOffset),
		faces, heatmapThreshold, nmsThreshold);
	getBox(faces);
}
*/
/*
void FaceDetector::postProcess(
        float* heatmap, float* scale, float* offset,
        std::vector<FaceInfo>& faces, float heatmapThreshold, float nmsThreshold){

	int heatmap_height = OUTPUT_HEIGHT/4;
	int heatmap_width = OUTPUT_WIDTH/4;
	int spacial_size = heatmap_height * heatmap_width;

	float* scale1 = scale + spacial_size;
	float* offset1 = offset + spacial_size;

	std::vector<int> heatmap_ids = filterHeatmap(heatmap, heatmap_height, heatmap_width, heatmapThreshold);

	std::vector<FaceInfo> faces_tmp;
	for (int i = 0; i < heatmap_ids.size()/2; i++){
        int id_h = heatmap_ids[2*i];
        int id_w = heatmap_ids[2*i+1];
        int index = id_h*heatmap_width + id_w;

        float s0 = std::exp(scale[index]) * 4;
        float s1= std::exp(scale1[index]) * 4;
        float o0 = offset[index];
        float o1= offset1[index];

        float x1 = std::max(0.0, (id_w + o1 + 0.5) * 4 - s1 / 2);
        float y1 = std::max(0.0, (id_h + o0 + 0.5) * 4 - s0 / 2);
        float x2 = 0, y2 = 0;
        x1 = std::min(x1, (float)d_w);
        y1 = std::min(y1, (float)d_h);
        x2 = std::min(x1 + s1, (float)d_w);
        y2 = std::min(y1 + s0, (float)d_h);

        FaceInfo facebox;
        facebox.x1 = x1*d_scale_w*scale_w;
        facebox.y1 = y1*d_scale_h*scale_h;
        facebox.x2 = x2*d_scale_w*scale_w;
        facebox.y2 = y2*d_scale_h*scale_h;
        facebox.score = heatmap[index];

        faces_tmp.push_back(facebox);
	}
	nms(faces_tmp, faces, nmsThreshold);
}
*/
/*
void FaceDetector::nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,
		float nmsThreshold){

	std::sort(input.begin(), input.end(),
		[](const FaceInfo& a, const FaceInfo& b){return a.score > b.score;});

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;

		output.push_back(input[i]);

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;
			float area1 = h1 * w1;

			float iou = inner_area / (area0 + area1 - inner_area);
			if (iou > nmsThreshold)
				merged[j] = 1;
		}

	}
}
*/
/*
void FaceDetector::dynamic_scale(float in_w, float in_h){
	d_h = (int)(std::ceil(in_h / 32) * 32);
	d_w = (int)(std::ceil(in_w / 32) * 32);

	d_scale_h = in_h/d_h ;
	d_scale_w = in_w/d_w ;
}

std::vector<int> FaceDetector::filterHeatmap(float* heatmap, int  h, int w, float thresh){
	std::vector<int> heatmap_ids;
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			if (heatmap[i*w + j] > thresh){
				heatmap_ids.push_back(i);
				heatmap_ids.push_back(j);
			}
		}
	}
	return heatmap_ids;
}

void FaceDetector::getBox(std::vector<FaceInfo>& faces){
	float w = 0, h = 0, maxSize = 0;
	float cenx, ceny;
	for (int i = 0; i < faces.size(); i++){
		w = faces[i].x2 - faces[i].x1;
		h = faces[i].y2 - faces[i].y1;

		maxSize = std::max(w, h);
		cenx = faces[i].x1 + w / 2;
		ceny = faces[i].y1 + h / 2;

		faces[i].x1 = std::max(cenx - maxSize / 2, 0.f);
		faces[i].y1 = std::max(ceny - maxSize / 2, 0.f);
		faces[i].x2 = std::min(cenx + maxSize / 2, image_w - 1.f);
		faces[i].y2 = std::min(ceny + maxSize / 2, image_h - 1.f);
	}
}
*/

BlazeFaceDetector::~BlazeFaceDetector()
{
    /*
    if(mModelBuffer != nullptr)
    {
        free(mModelBuffer);
        mModelBuffer = nullptr;
    }
    */
}

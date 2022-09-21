
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#include "BlazeFaceDetector.h"
#include "BlazeFaceUtil.h"
#include "Utils.hpp"

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

/////////////////////////////////////////////////////////////////////////////////////////

BlazeFaceDetector::BlazeFaceDetector(float scoreThreshold, float iouThreshold)
{
    //self.type = type
    mScoreThreshold = scoreThreshold;
    mIouThreshold = iouThreshold;
    mSigScoreTh = log(mScoreThreshold / (1.0 - mScoreThreshold));

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

bool BlazeFaceDetector::DetectFaces(const Mat& srcImage)
{
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
    cout << "-----------------------------------------------" << endl;
    for(int i=0; i<indexScoreCds.size(); i++ )
    {
        cout << indexScoreCds[i].index << ", " << indexScoreCds[i].score << endl;
    }
    cout << "-----------------------------------------------" << endl;

     /*

# Extract information of filtered detections
boxes, keypoints = self.extractDetections(output0, goodDetectionsIndices)

# Filter results with non-maximum suppression
detectionResults = self.filterWithNonMaxSupression(boxes, keypoints, scores)

# Update fps calculator
self.updateFps()

return detectionResults
     */
    return true;
}

// Cds for Candidates
void BlazeFaceDetector::filterDetections(vector<FaceIndexScore>& indexScoreCds)
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

void BlazeFaceDetector::extractDetections(const vector<int>& goodIndices)
{
    // bounding box and six key points
    float* out0Ptr = mInterpreter->typed_output_tensor<float>(0);
    //float boxKeyPt[][16]

    for(int anchorID: goodIndices)
    {
        const Anchor& anchor = mAnchors[anchorID];

        float* pBoxKeyPts = out0Ptr + anchorID * 16;
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
            
            // store lx, ly in some container.
        }
    }
}
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

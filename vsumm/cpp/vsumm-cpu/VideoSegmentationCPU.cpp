#include "VideoSegmentationCPU.h"
#include "Results.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iomanip>
#include <iostream>

using namespace cv;

vector<Mat> VideoSegmentationCPU::readKeepFrames()
{
	vector<Mat> allFrames;
	Mat frame;
	VideoCapture reader(this->videoName);

	if(!reader.isOpened())
		return vector<Mat>();

	double fps = reader.get(CV_CAP_PROP_FPS);
	double frameCount = reader.get(CV_CAP_PROP_FRAME_COUNT);

	int framesRead = 1;
	int nextFrame = 0;
	while(framesRead <= (int)(frameCount/fps))
	{
		if(!reader.read(frame))
			return vector<Mat>();
		
		double totalSize = (frame.rows*frame.cols*3)*(frameCount/fps);
		try{
			allFrames.reserve(totalSize);
		}catch(std::bad_alloc &e){
			cout << "no memory available to keep the frames" << endl;
			return vector<Mat>();
		}

		allFrames.push_back(frame);
		framesRead++;
		nextFrame += (int)fps;

		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
	}
	reader.release();
	return allFrames;
}

int VideoSegmentationCPU::readSaveFrames(string& dirName)
{
	Mat frame;
	VideoCapture reader(this->videoName);

	if(!reader.isOpened())
		return 0;

	double fps = reader.get(CV_CAP_PROP_FPS);
	double frameCount = reader.get(CV_CAP_PROP_FRAME_COUNT);

	int framesSaved = 1;
	int nextFrame = 0;
	while(framesSaved <= (int)(frameCount / fps))
	{
		if(!reader.read(frame))
			return 0;
		
		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << framesSaved;

		string name = dirName+"/frame-"+out.str()+".jpg";

		imwrite(name, frame);
		framesSaved++;
		nextFrame += (int)fps;

		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
	}

	reader.release();
	return 1;
}

int VideoSegmentationCPU::readDiscardFrames()
{
	Mat frame;
	VideoCapture reader(this->videoName);

	if(!reader.isOpened())
		return 0;

	double fps = reader.get(CV_CAP_PROP_FPS);
	double frameCount = reader.get(CV_CAP_PROP_FRAME_COUNT);

	int framesRead = 1;
	int nextFrame = 0;
	while(framesRead <= (int)(frameCount / fps))
	{
		if(!reader.read(frame))
			return 0;
		
		framesRead++;
		nextFrame += (int)fps;
		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
	}
	reader.release();
	return 1;
}

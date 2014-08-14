#include "DecoderCPU.h"
#include "Results.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iomanip>
#include <iostream>

using namespace cv;

int DecoderCPU::saveFrames(string& videoName, string& dirName)
{
	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		return 0;

	double fps = reader.get(CV_CAP_PROP_FPS);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);

	int count = 1;
	int nextFrame = 0;
	while(count <= (int)(frameQnt / fps))
	{
		if(!reader.read(frame))
			break;

		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count;

		string name = dirName+"\\frame-"+out.str()+".jpg";

		imwrite(name, frame);
		nextFrame += (int)fps;

		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
		count++;
	}

	reader.release();

	return 1;
}

int DecoderCPU::decodeWithoutSave(string& videoName, string& dirName)
{
	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		return 0;

	double fps = reader.get(CV_CAP_PROP_FPS);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);

	int count = 1;
	int nextFrame = 0;
	while(count <= (int)(frameQnt / fps))
	{
		if(!reader.read(frame))
			break;
		nextFrame += (int)fps;

		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
		count++;
	}

	reader.release();
	return 1;
}

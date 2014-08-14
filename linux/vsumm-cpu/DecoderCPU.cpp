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
	cv::TickMeter timeLocal; double time = 0.0;

	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		return 0;

	double fps = reader.get(CV_CAP_PROP_FPS);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);

	fps = 25;

	int count = 1;
	int nextFrame = 0;
	while(count <= (int)(frameQnt / fps))
	{
		timeLocal.reset(); timeLocal.start();
		if(!reader.read(frame))
			break;
		timeLocal.stop(); time += timeLocal.getTimeSec();

		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count;

		string name = dirName+"/frame-"+out.str()+".jpg";

		imwrite(name, frame);
		nextFrame += (int)fps;

		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
		count++;
	}

	reader.release();

	Results *result;
	result = Results::getInstance();
	result->setDecodeWithoutWrite(time);

	return 1;
}

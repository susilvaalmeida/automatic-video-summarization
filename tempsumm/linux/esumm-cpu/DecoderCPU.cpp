#include "DecoderCPU.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iomanip>
#include <iostream>

#include "Results.h"

using namespace cv;

int DecoderCPU::saveFrames(string& videoName, string& dirName)
{
	double time_without_write = 0.0;
	cv::TickMeter localTime;

	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		return 0;

	double fps = reader.get(CV_CAP_PROP_FPS);

	//cout << fps << endl;
	//fps = 25;

	int count = 1;
	//int nextFrame = 0;
	while(true)
	{
		localTime.reset(); localTime.start();
		if(!reader.read(frame))
			break;
		localTime.stop(); time_without_write += localTime.getTimeSec();

		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frame);

		count++;
	}

	reader.release();

	Results *result;
	result = Results::getInstance();
	result->setDecodeWithoutWrite(time_without_write);


	return 1;
}

#include "DecoderGPU.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iomanip>
#include <iostream>

#include "Results.h"

using namespace cv;

int DecoderGPU::saveFramesGPU(string& videoName, string& dirName)
{
	double time_without_write = 0.0;
	double time_gpu_cp = 0.0;
	cv::TickMeter localTime, localTime2, globalTime;

	globalTime.reset(); globalTime.start();

	gpu::GpuMat frame;
	gpu::VideoReader_GPU reader(videoName);

	if(!reader.isOpened())
		return 0;

	VideoCapture readerCPU(videoName);
	//double fps = readerCPU.get(CV_CAP_PROP_FPS);
	double frameQnt = readerCPU.get(CV_CAP_PROP_FRAME_COUNT);

	int count = 1;
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
		//cout << name << endl;

		localTime2.reset(); localTime2.start();
		cv::Mat frameSaida(frame);
		localTime2.stop(); time_gpu_cp += localTime2.getTimeSec();

		imwrite(name, frameSaida);

		count++;
	}
	reader.close();


	Mat frameCPU;
	readerCPU.set(CV_CAP_PROP_POS_FRAMES, count);

	while(count < frameQnt)
	{
		localTime.reset(); localTime.start();
		if(!readerCPU.read(frameCPU))
			break;
		localTime.stop(); time_without_write += localTime.getTimeSec();


		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frameCPU);

		count++;
	}
	readerCPU.release();

	globalTime.stop();


	Results *result;
	result = Results::getInstance();
	result->setDecodeWithoutWrite(time_without_write);
	result->setDecodeWithoutGpuCopy(globalTime.getTimeSec() - time_gpu_cp);

	return 1;
}

int DecoderGPU::saveFrames(string& videoName, string& dirName)
{
	double time_without_write = 0.0;
	cv::TickMeter localTime;

	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		return 0;

	//double fps = reader.get(CV_CAP_PROP_FPS);

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



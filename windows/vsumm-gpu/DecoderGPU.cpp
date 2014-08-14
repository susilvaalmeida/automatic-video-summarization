#include "DecoderGPU.h"
#include "Results.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iomanip>
#include <iostream>

using namespace cv;

int DecoderGPU::saveFramesGPU(string& videoName, string& dirName)
{
	gpu::GpuMat frame;
	gpu::VideoReader_GPU reader(videoName);

	if(!reader.isOpened())
		return 0;

	VideoCapture readerCPU(videoName);
	double fps = readerCPU.get(CV_CAP_PROP_FPS);
	double frameQnt = readerCPU.get(CV_CAP_PROP_FRAME_COUNT);


	int count = 0;
	int nextFrame = 1;
	while(true)
	{
		if(!reader.read(frame))
			break;

		if(count%(int)fps==0)
		{
			stringstream out;
			out.fill('0');
			out << std::right << std::setw(6) << nextFrame;

			string name = dirName+"/frame-"+out.str()+".jpg";
			cv::Mat frameSaida(frame);
			imwrite(name, frameSaida);
			nextFrame ++;
		}
		count++;
	}
	reader.close();


	Mat frameCPU;
	readerCPU.set(CV_CAP_PROP_POS_FRAMES, count);
	int count2 = nextFrame-1;

	nextFrame = count;
	while(count < frameQnt)
	{
		if(!readerCPU.read(frameCPU))
			break;
		
		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count2;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frameCPU);

		nextFrame += (int)fps;

		if(count < frameQnt - 1)
		{
			readerCPU.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
		}
		count++;
		count2++;
	}
	readerCPU.release();

	
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

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frame);

		nextFrame += (int)fps;
		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);

		count++;
	}

	reader.release();
	return 1;
}

vector<HistogramGPU> DecoderGPU::saveFramesAndComputeHistGPU(string& videoName, string& dirName)
{
	vector<HistogramGPU> hists;

	gpu::GpuMat frame;
	gpu::VideoReader_GPU reader(videoName);

	VideoCapture readerCPU(videoName);
	double fps = readerCPU.get(CV_CAP_PROP_FPS);
	double frameQnt = readerCPU.get(CV_CAP_PROP_FRAME_COUNT);

	int count = 0;
	int nextFrame = 1;
	while(true)
	{
		if(!reader.read(frame))
			break;

		if(count%(int)fps==0)
		{
			stringstream out;
			out.fill('0');
			out << std::right << std::setw(6) << nextFrame;

			string name = dirName+"/frame-"+out.str()+".jpg";

			HistogramGPU hist = FeaturesGPU::computeOneHist(frame, nextFrame-1, frame.rows, frame.cols);
			if(hist.getHistogram().size() > 0)
				hists.push_back(hist);

			cv::Mat frameSaida(frame);
			imwrite(name, frameSaida);
			nextFrame ++;
		}
		count++;
	}
	reader.close();

	if(nextFrame < frameQnt / fps)
	{
	Mat frameCPU;
	readerCPU.set(CV_CAP_PROP_POS_FRAMES, count);
	int count2 = nextFrame-1;

	nextFrame = count;
	while(count2 <= frameQnt/fps)
	{
		if(!readerCPU.read(frameCPU))
			break;

		cv::gpu::GpuMat frameGpu(frameCPU);
		HistogramGPU hist = FeaturesGPU::computeOneHist(frameGpu, count2-1, frameGpu.rows, frameGpu.cols);
		if(hist.getHistogram().size() > 0)
			hists.push_back(hist);

		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count2;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frameCPU);
	
		nextFrame += (int)fps;

		if(count2 < frameQnt/fps)
		{
			readerCPU.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
		}
		count++;
		count2++;
	}
	readerCPU.release();

	}
	
	return hists;
}

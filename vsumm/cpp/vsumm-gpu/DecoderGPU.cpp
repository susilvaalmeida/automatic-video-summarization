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
	double time_without_write = 0.0;
	double time_gpu_cp = 0.0;
	cv::TickMeter localTime, localTime2, globalTime;

	globalTime.reset(); globalTime.start();

	gpu::GpuMat frame;
	gpu::VideoReader_GPU reader(videoName);

	if(!reader.isOpened())
		return 0;

	VideoCapture readerCPU(videoName);
	double fps = readerCPU.get(CV_CAP_PROP_FPS);
	double frameQnt = readerCPU.get(CV_CAP_PROP_FRAME_COUNT);


	//if(fps < 20 || fps > 60)
	fps = 25;
	//cout << "fps: " << fps << endl;

	int count = 0;
	int nextFrame = 1;
	while(true)
	{
		localTime.reset(); localTime.start();
		if(!reader.read(frame))
			break;
		localTime.stop(); time_without_write += localTime.getTimeSec();

		if(count%(int)fps==0)
		{
			stringstream out;
			out.fill('0');
			out << std::right << std::setw(6) << nextFrame;

			string name = dirName+"/frame-"+out.str()+".jpg";

			localTime2.reset(); localTime2.start();
			cv::Mat frameSaida(frame);
			localTime2.stop(); time_gpu_cp += localTime2.getTimeSec();

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
		localTime.reset(); localTime.start();
		if(!readerCPU.read(frameCPU))
			break;
		localTime.stop(); time_without_write += localTime.getTimeSec();

		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count2;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frameCPU);

		nextFrame += (int)fps;

		if(count < frameQnt - 1)
		{
			//cv::TickMeter time; time.reset(); time.start();
			readerCPU.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
			//time.stop();
			//cout << nextFrame << "-" << time.getTimeSec() << endl;
		}
		count++;
		count2++;
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

	double fps = reader.get(CV_CAP_PROP_FPS);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);

	fps = 25;

	int count = 1;
	int nextFrame = 0;
	while(count <= (int)(frameQnt / fps))
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

		nextFrame += (int)fps;
		reader.set(CV_CAP_PROP_POS_FRAMES, nextFrame);

		count++;
	}

	Results *result;
	result = Results::getInstance();
	result->setDecodeWithoutWrite(time_without_write);

	reader.release();
	return 1;
}

vector<HistogramGPU> DecoderGPU::saveFramesAndComputeHistGPU(string& videoName, string& dirName)
{
	double timeParallel = 0.0;
	cv::TickMeter localTime;

	vector<HistogramGPU> hists;

	gpu::GpuMat frame;
	gpu::VideoReader_GPU reader(videoName);

	VideoCapture readerCPU(videoName);
	double fps = readerCPU.get(CV_CAP_PROP_FPS);
	double frameQnt = readerCPU.get(CV_CAP_PROP_FRAME_COUNT);


	//if(fps < 20 || fps > 60)
	fps = 25;
	//cout << "fps: " << fps << endl;

	int count = 0;
	int nextFrame = 1;
	while(true)
	{

		if(!reader.read(frame))
			break;


		if(count%(int)fps==0)
		{
			localTime.reset(); localTime.start();
			//cout << "decoding gpu : count " <<  nextFrame << " fps-" << fps<< endl;

			stringstream out;
			out.fill('0');
			out << std::right << std::setw(6) << nextFrame;

			string name = dirName+"/frame-"+out.str()+".jpg";
			//cout << name << endl;

			HistogramGPU hist = FeaturesGPU::computeOneHist(frame, nextFrame-1, frame.rows, frame.cols);
			if(hist.getHistogram().size() > 0)
				hists.push_back(hist);

			//cout << hists.size() << endl;

			cv::Mat frameSaida(frame);
			localTime.stop(); timeParallel += localTime.getTimeSec();
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
	while(count2 <= frameQnt/fps)
	{
		localTime.reset(); localTime.start();

		if(!readerCPU.read(frameCPU))
			break;

		cv::gpu::GpuMat frameGpu(frameCPU);
		HistogramGPU hist = FeaturesGPU::computeOneHist(frameGpu, count2-1, frameGpu.rows, frameGpu.cols);
		if(hist.getHistogram().size() > 0)
			hists.push_back(hist);

		//cout << hists.size() << endl;
		localTime.stop(); timeParallel += localTime.getTimeSec();

		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count2;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frameCPU);
		//cout << name << endl;

		nextFrame += (int)fps;

		if(count2 < frameQnt/fps)
		{
			//cv::TickMeter time; time.reset(); time.start();
			readerCPU.set(CV_CAP_PROP_POS_FRAMES, nextFrame);
			//time.stop();
			//cout << nextFrame << "-" << time.getTimeSec() << endl;
		}
		count++;
		count2++;
	}
	readerCPU.release();


	Results *result;
	result = Results::getInstance();
	result->setFeatExtractionParallelPart(timeParallel);

	return hists;
}

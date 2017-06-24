#include "DecoderMULT.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iomanip>
#include <iostream>
#include <thread>
#include <omp.h>

#include "Results.h"
#include "Defines.h"


using namespace cv;

int DecoderMULT::saveFrames(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = reader.get(CV_CAP_PROP_FPS);
	fps = 25;
	reader.release();

	vector<int> indexes;
	int idFrame = 0;
	for(int i=0; (int)indexes.size()<=(int)(frameQnt/fps); i++)
	{
		indexes.push_back(idFrame);
		idFrame += fps;
	}

	int max_num_threads = omp_get_max_threads();
	cout << "using " << max_num_threads << " threads" << endl;

	int framesPerThread = std::floor(indexes.size() / max_num_threads);

	thread threads[max_num_threads];
	int count = 0;

	//timeLocal.reset(); timeLocal.start();
	for(int i=0; i<max_num_threads; i++)
	{
		if(i==max_num_threads-1)
		{
			vector<int> indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.end());
			threads[i] = thread(DecoderMULT::save_func, std::ref(videoName), std::ref(dirName), count, indexes_aux);
		}
		else
		{
			vector<int> indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.begin()+((i+1)*framesPerThread)+1);
			threads[i] = thread(DecoderMULT::save_func, std::ref(videoName), std::ref(dirName), count, indexes_aux);
		}
		count += framesPerThread;
	}

	for(int i=0; i<max_num_threads; i++)
		threads[i].join();

	//timeLocal.stop();
	//result->setDecode(timeLocal.getTimeSec());
	//cout << "decoding: " << timeLocal.getTimeSec();

	return 1;
}

int DecoderMULT::saveFramesOMP(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = reader.get(CV_CAP_PROP_FPS);
	fps = 25;
	reader.release();

	vector<int> indexes;
	int idFrame = 0;
	for(int i=0; (int)indexes.size()<=(int)(frameQnt/fps); i++)
	{
		indexes.push_back(idFrame);
		idFrame += fps;
	}

	int max_num_threads = omp_get_max_threads();
	int framesPerThread = std::floor(indexes.size() / max_num_threads);

	setNumThreads(max_num_threads);

#pragma omp parallel for
	for(int i=0; i<max_num_threads; i++)
	{
		int count = i*framesPerThread;
		vector<int> indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.begin()+((i+1)*framesPerThread)+1);

		if(i == max_num_threads-1)
			indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.end());
		save_func(videoName, dirName, count, indexes_aux);
	}
}

void DecoderMULT::save_func(string& videoName, string& dirName, int start, vector<int> vetor)
{
	//cout << omp_get_thread_num() << "-" << start << endl;
	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		std::cout << "erro reader" << endl;

	int count = start;

	reader.set(CV_CAP_PROP_POS_FRAMES, vetor[0]);

	reader.read(frame);

	for(int i=1; i<(int)vetor.size(); i++)
	{
		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count+1;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frame);

		//cv::TickMeter time; time.reset(); time.start();
		reader.set(CV_CAP_PROP_POS_FRAMES, vetor[i]);
		//time.stop();
		//cout << vetor[i] << "-" << time.getTimeSec() << endl;

		if(i < vetor.size()-1)
			reader.read(frame);

		count++;
	}
	reader.release();
}

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

int DecoderMULT::saveFramesTHR(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = reader.get(CV_CAP_PROP_FPS);
	reader.release();

	vector<int> indexes;
	int idFrame = 0;
	for(int i=0; (int)indexes.size()<=(int)(frameQnt/fps); i++)
	{
		indexes.push_back(idFrame);
		idFrame += (int)fps;
	}

	const int max_num_threads = omp_get_max_threads();
	int framesPerThread = (int)std::floor((int)indexes.size() / max_num_threads);

	vector<thread> threads(max_num_threads);
	int count = 0;

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

	return 1;
}

int DecoderMULT::decodeWithouSaveTHR(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = reader.get(CV_CAP_PROP_FPS);
	reader.release();

	vector<int> indexes;
	int idFrame = 0;
	for(int i=0; (int)indexes.size()<=(int)(frameQnt/fps); i++)
	{
		indexes.push_back(idFrame);
		idFrame += (int)fps;
	}

	const int max_num_threads = omp_get_max_threads();
	int framesPerThread = (int)std::floor((int)indexes.size() / max_num_threads);

	vector<thread> threads(max_num_threads);
	int count = 0;

	for(int i=0; i<max_num_threads; i++)
	{
		if(i==max_num_threads-1)
		{
			vector<int> indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.end());
			threads[i] = thread(DecoderMULT::decode_without_save, std::ref(videoName), std::ref(dirName), count, indexes_aux);
		}
		else
		{
			vector<int> indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.begin()+((i+1)*framesPerThread)+1);
			threads[i] = thread(DecoderMULT::decode_without_save, std::ref(videoName), std::ref(dirName), count, indexes_aux);
		}
		count += framesPerThread;
	}

	for(int i=0; i<max_num_threads; i++)
		threads[i].join();

	return 1;
}

int DecoderMULT::saveFramesOMP(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = reader.get(CV_CAP_PROP_FPS);
	reader.release();

	vector<int> indexes;
	int idFrame = 0;
	for(int i=0; (int)indexes.size()<=(int)(frameQnt/fps); i++)
	{
		indexes.push_back(idFrame);
		idFrame += (int)fps;
	}

	int max_num_threads = omp_get_max_threads();
	int framesPerThread = (int)std::floor(indexes.size() / max_num_threads);

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
	return 1;
}

int DecoderMULT::decodeWithouSaveOMP(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = reader.get(CV_CAP_PROP_FPS);
	reader.release();

	vector<int> indexes;
	int idFrame = 0;
	for(int i=0; (int)indexes.size()<=(int)(frameQnt/fps); i++)
	{
		indexes.push_back(idFrame);
		idFrame += (int)fps;
	}

	int max_num_threads = omp_get_max_threads();
	int framesPerThread = (int)std::floor(indexes.size() / max_num_threads);

	setNumThreads(max_num_threads);

	#pragma omp parallel for
	for(int i=0; i<max_num_threads; i++)
	{
		int count = i*framesPerThread;
		vector<int> indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.begin()+((i+1)*framesPerThread)+1);

		if(i == max_num_threads-1)
			indexes_aux = vector<int>(indexes.begin()+i*framesPerThread, indexes.end());
		decode_without_save(videoName, dirName, count, indexes_aux);
	}
	return 1;
}

void DecoderMULT::save_func(string& videoName, string& dirName, int start, vector<int> vetor)
{
	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		std::cout << "erro reader" << endl;

	int count = start;

	bool set = reader.set(CV_CAP_PROP_POS_FRAMES, vetor[0]);
	if(!set)
		cout << "erro set" << endl;

	reader.read(frame);

	for(int i=1; i<(int)vetor.size(); i++)
	{
		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count+1;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frame);

		set = reader.set(CV_CAP_PROP_POS_FRAMES, vetor[i]);
		if(!set)
			cout << "erro set" << endl;
		
		if(!reader.read(frame))
			break;

		count++;
	}
	reader.release();
}

void DecoderMULT::decode_without_save(string& videoName, string& dirName, int start, vector<int> vetor)
{
	Mat frame;
	VideoCapture reader(videoName);

	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	if(!reader.isOpened())
		std::cout << "erro reader" << endl;

	int count = start;

	bool set = reader.set(CV_CAP_PROP_POS_FRAMES, vetor[0]);
	if(!set)
		cout << "erro set" << endl;

	reader.read(frame);

	for(int i=1; i<(int)vetor.size(); i++)
	{
		if(vetor[i] >= frameQnt)
			break;

		//cout << vetor[i] << endl;
		set = reader.set(CV_CAP_PROP_POS_FRAMES, vetor[i]);
		if(!set)
			cout << "erro set" << endl;
		if(!reader.read(frame))
			break;
		count++;
	}
	reader.release();
}
#include "DecoderMULT.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iomanip>
#include <iostream>
#include <omp.h>
#include <thread>

using namespace cv;

int DecoderMULT::saveFrames(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	reader.release();

	int max_num_threads = omp_get_max_threads();

	int id = 0;
	int jump = frameQnt / max_num_threads;

	vector<thread> threads;


	for(int i=0; i<max_num_threads; i++)
	{
		if(i==max_num_threads-1)
			threads.push_back(thread(DecoderMULT::save_func, std::ref(videoName), std::ref(dirName), id, frameQnt));
		else
			threads.push_back(thread(DecoderMULT::save_func, std::ref(videoName), std::ref(dirName), id, id+jump));
		id+=jump;
	}

	for(int i=0; i<max_num_threads; i++)
		threads[i].join();

	return 1;
}

int DecoderMULT::saveFramesOMP(string& videoName, string& dirName)
{
	VideoCapture reader(videoName);
	double frameQnt = reader.get(CV_CAP_PROP_FRAME_COUNT);
	reader.release();

	int max_num_threads = omp_get_max_threads();

	setNumThreads(max_num_threads);

	int id = 0;
	int jump = frameQnt / max_num_threads;

#pragma omp parallel for
	for(int i=0; i<max_num_threads; i++)
	{
		if(i==max_num_threads-1)
			save_func(videoName, dirName, i*jump, frameQnt);
		else
			save_func(videoName, dirName, i*jump, (i*jump)+jump);
	}
}

void DecoderMULT::save_func(string& videoName, string& dirName, int start, int end)
{
	Mat frame;
	VideoCapture reader(videoName);

	if(!reader.isOpened())
		std::cout << "erro reader" << endl;

	int count = start;

	reader.set(CV_CAP_PROP_POS_FRAMES, start);

	reader.read(frame);

	for(int i=start+1; i<end; i++)
	{
		stringstream out;
		out.fill('0');
		out << std::right << std::setw(6) << count+1;

		string name = dirName+"/frame-"+out.str()+".jpg";
		imwrite(name, frame);

		reader.read(frame);

		count++;
	}
	reader.release();

}


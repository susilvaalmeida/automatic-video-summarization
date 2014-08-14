#include "HistogramMULT.h"
#include "Defines.h"
#include "Results.h"

#include <iostream>

#include <thread>
#include <mutex>
#include <omp.h>

using namespace std;

HistogramMULT::HistogramMULT(cv::Mat& img, int bins, int idFrame)
: idFrame(idFrame), bins(bins)
{
	channels = img.channels();
	freqTotal = img.rows * img.cols;
	hist.resize(bins*channels, 0.0);

	for(int i=0; i<img.rows; i++)
	{
		for(int j=0; j<img.cols; j++)
		{
			int r = (int)img.at<cv::Vec3b>(i,j)[2];
			int g = (int)img.at<cv::Vec3b>(i,j)[1];
			int b = (int)img.at<cv::Vec3b>(i,j)[0];

			hist[r / bins]++;
			hist[(g / bins) + bins]++;
			hist[(b / bins) + 2*bins]++;
		}
	}
}

float HistogramMULT::stdDev()
{
	float var = 0;
	float mean = 0;

	vector<float> hist_aux = hist;

#pragma omp parallel for
	for(int i=0; i<(int)hist.size(); i++)
		hist_aux[i] = getHistPosNorm(i);

	for(int i=0; i<(int)hist.size(); i++)
		mean += hist_aux[i];
	mean /= hist.size();

	for(int i=0; i<(int)hist.size(); i++)
		var += pow(hist_aux[i]-mean, 2);

	return sqrt(var/hist.size());
}

vector<float> HistogramMULT::getHistogramNorm()
{
	vector<float> norm;
	for(int i=0; i<BINS; i++)
		norm.push_back(getHistPosNorm(i));
	return norm;
}

bool sortfunction1(HistogramMULT h1, HistogramMULT h2)
{
	return h1.getIdFrame() < h2.getIdFrame();
}

FeaturesMULT::FeaturesMULT(vector<string>& frames, string type)
{
	cv::TickMeter timeLocal;

	timeLocal.reset(); timeLocal.start();
	int max_num_threads = omp_get_max_threads();

	allHist.clear();

	int id = 0;
	int jump = frames.size() / max_num_threads;

	if(type == "thr")
	{
		vector<thread> threads;

		for(int i=0; i<max_num_threads; i++)
		{
			if(i==max_num_threads-1)
				threads.push_back(thread(&FeaturesMULT::computeHist, this, std::ref(frames), id, (int)frames.size()));
			else
				threads.push_back(thread(&FeaturesMULT::computeHist, this, std::ref(frames), id, id+jump));
			id+=jump;
		}

		for(int i=0; i<max_num_threads; i++)
			threads[i].join();

	}
	else if(type == "omp")
	{
		int max_num_threads = omp_get_max_threads();
		int framesPerThread = frames.size() / max_num_threads;

		omp_set_num_threads(max_num_threads);

		int id = 0;
		#pragma omp parallel for
		for(int i=0; i<max_num_threads; i++)
		{
			computeHistOMP(frames, i*framesPerThread, (i*framesPerThread)+framesPerThread);
			//cout << i*framesPerThread << "," << (i*framesPerThread)+framesPerThread << endl;
			//id+=framesPerThread;
		}
	}

	timeLocal.stop();

	Results *result;
	result = Results::getInstance();
	result->setFeatExtractionParallelPart(timeLocal.getTimeSec());

	sort(allHist.begin(), allHist.end(), sortfunction1);
}

std::mutex values_mutex;
void FeaturesMULT::computeHist(vector<string>& frames, int start, int end)
{
	for(int i=start; i<end; i++)
	{
		cv::Mat img = cv::imread(frames[i]);

		float mean = HistogramMULT(img, BINS, i).stdDev();

		//considera apenas frames com determinado desvio padrao
		if(mean < 0.23)
		{
			//converte para HSV
			cv::Mat imgHSV;
			cv::cvtColor(img, imgHSV, CV_RGB2HSV);

			HistogramMULT hist_aux = HistogramMULT(imgHSV, BINS, i);

			vector<float> vaux = hist_aux.getHistogram();
			hist_aux.setHistogram(vector<float>(vaux.begin(), vaux.begin()+BINS));

			values_mutex.lock();
			allHist.push_back(hist_aux);
			values_mutex.unlock();
		}
	}
}

void FeaturesMULT::computeHistOMP(vector<string>& frames, int start, int end)
{
	for(int i=start; i<end; i++)
	{
		cv::Mat img = cv::imread(frames[i]);

		float mean = HistogramMULT(img, BINS, i).stdDev();

		//considera apenas frames com determinado desvio padrao
		if(mean < 0.23)
		{
			//converte para HSV
			cv::Mat imgHSV;
			cv::cvtColor(img, imgHSV, CV_RGB2HSV);

			HistogramMULT hist_aux = HistogramMULT(imgHSV, BINS, i);

			vector<float> vaux = hist_aux.getHistogram();
			hist_aux.setHistogram(vector<float>(vaux.begin(), vaux.begin()+BINS));

#pragma omp critical
			{
				allHist.push_back(hist_aux);
			}
		}
	}
}

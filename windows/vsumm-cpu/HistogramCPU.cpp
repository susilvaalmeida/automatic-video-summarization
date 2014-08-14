#include "HistogramCPU.h"
#include "Defines.h"
#include "Results.h"

#include <iostream>

HistogramCPU::HistogramCPU(cv::Mat& img, int bins, int idFrame)
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

float HistogramCPU::stdDev()
{
	float var = 0;
	float mean = 0;

	vector<float> hist_aux = hist;

	for(int i=0; i<(int)hist.size(); i++)
		hist_aux[i] = getHistPosNorm(i);

	for(int i=0; i<(int)hist.size(); i++)
		mean += hist_aux[i];

	mean /= hist.size();

	for(int i=0; i<(int)hist.size(); i++)
		var += pow(hist_aux[i]-mean, 2);
	return sqrt(var/hist.size());
}

vector<float> HistogramCPU::getHistogramNorm()
{
	vector<float> norm;
	for(int i=0; i<BINS; i++)
		norm.push_back(getHistPosNorm(i));
	return norm;
}

vector<HistogramCPU> FeaturesCPU::computeAllHist(vector<string> frames)
{
	vector<HistogramCPU> hists;
	for(int i=0; i<(int)frames.size(); i++)
	{
		cv::Mat img = cv::imread(frames[i]);
		float mean = HistogramCPU(img, BINS, i).stdDev();

		//considera apenas frames com determinado desvio padrao
		if(mean < 0.23)
		{
			//converte para HSV
			cv::Mat imgHSV;
			cv::cvtColor(img, imgHSV, CV_RGB2HSV);

			HistogramCPU hist_aux = HistogramCPU(imgHSV, BINS, i);

			vector<float> vaux = hist_aux.getHistogram();
			hist_aux.setHistogram(vector<float>(vaux.begin(), vaux.begin()+BINS));
			hists.push_back(hist_aux);
		}
	}


	return hists;
}

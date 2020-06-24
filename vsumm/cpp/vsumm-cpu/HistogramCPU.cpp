#include "HistogramCPU.h"
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

HistogramCPU::HistogramCPU(cv::Mat& img, int idFrame)
: idFrame(idFrame)
{
	int bins = 16;
	float range[] = { 0, 256 } ;
  	const float* histRange = { range };

	vector<cv::Mat> bgr_planes;
	cv::split(img, bgr_planes);

	cv::Mat b_hist, g_hist, r_hist;
	cv::calcHist(&bgr_planes[0],1,0,cv::Mat(),b_hist,1,&bins,&histRange);
	cv::calcHist(&bgr_planes[1],1,0,cv::Mat(),g_hist,1,&bins,&histRange);
	cv::calcHist(&bgr_planes[2],1,0,cv::Mat(),r_hist,1,&bins,&histRange);

	hist.resize(bins*3, 0.0);
	for(int i=0;i<b_hist.rows;i++)
	{
		hist[i] = r_hist.at<float>(i);
		hist[i+bins] = g_hist.at<float>(i);
		hist[i+(2*bins)] = b_hist.at<float>(i);
	}
}

float HistogramCPU::stdDev()
{
	hist = getHistogramNorm();
	cv::Mat mean;
	cv::Mat stddev;
	cv::meanStdDev(hist,mean,stddev);
	return stddev.at<double>(0,0);
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
		//float mean = HistogramCPU(img, BINS, i+1).stdDev();
		float mean = HistogramCPU(img, i+1).stdDev();

		if(mean < 0.23)
		{
			cv::Mat imgHSV;
			cv::cvtColor(img, imgHSV, CV_RGB2HSV);

			//HistogramCPU hist_aux = HistogramCPU(imgHSV, BINS, i+1);
			HistogramCPU hist_aux = HistogramCPU(imgHSV, i+1);

			vector<float> vaux = hist_aux.getHistogram();
			hist_aux.setHistogram(vector<float>(vaux.begin(), vaux.begin()+BINS));
			hists.push_back(hist_aux);
		}
	}
	return hists;
}

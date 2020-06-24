#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#define BINS 16
using namespace std;

class HistogramCPU {
public:
	HistogramCPU(cv::Mat& img, int bins, int idFrame);
	HistogramCPU(cv::Mat& img, int idFrame);

	vector<float> getHistogram(){ return this->hist; }
	void setHistogram(vector<float> hist){ hist.assign(hist.begin(), hist.end()); }

	vector<float> getHistogramNorm();

	int getIdFrame() const { return idFrame; }
	void setIdFrame(int idFrame) { this->idFrame = idFrame; }

	int getBins() const { return bins; }
	void setBins(int bins) { this->bins = bins; }

	int getChannels() const { return channels; }
	void setChannels(int channels) { this->channels = channels; }

	int getFreqTotal() const { return freqTotal; }
	void setFreqTotal(int freqTotal) { this->freqTotal = freqTotal; }

	float getHistPos(int pos){ return this->hist[pos]; }
	float getHistPosNorm(int pos){ return this->hist[pos] / freqTotal; }

	float stdDev();

private:
	int idFrame;
	vector<float> hist;
	int channels;
	int freqTotal;
	int bins;
};

class FeaturesCPU
{
public:
	static vector<HistogramCPU> computeAllHist(vector<string> frames);
};

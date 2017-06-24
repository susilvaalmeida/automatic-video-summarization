#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/gpu/gpu.hpp>
using namespace std;

class HistogramGPU {
public:
	HistogramGPU(vector<float> hist, int bins, int idFrame, int chan, int freq);
	HistogramGPU();

	vector<float> getHistogram(){ return this->hist; }
	void setHistogram(vector<float> hist){ this->hist = hist; }

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

	//float stdDev();

private:
	int idFrame;
	vector<float> hist;
	int channels;
	int freqTotal;
	int bins;
};

class FeaturesGPU
{
public:
	static vector<HistogramGPU> computeAllHist(unsigned char* images, int frameInicial, int qntFrames, int rows, int cols);
	static HistogramGPU computeOneHist(cv::gpu::GpuMat img, int idFrame, int rows, int cols);
};

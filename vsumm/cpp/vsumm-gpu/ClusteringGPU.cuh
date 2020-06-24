#pragma once

#include "HistogramGPU.cuh"

class ClusteringGPU {
public:
	ClusteringGPU(vector<HistogramGPU> hists): features(hists){};

	void kmeansGPU();
	void findKeyframes();
	void removeSimilarKeyframes();

	vector<HistogramGPU> getKeyframes() const { return keyframes; }

private:
	void estimateK();
	//float euclidianDist(HistogramGPU h1, HistogramGPU h2);
	float euclidianDist(HistogramGPU h1, vector<float> cluster);
	float euclidianDist(vector<float> h1, vector<float> h2);

	//int findNearestCluster(HistogramGPU hist);

	vector<HistogramGPU> features;
	vector<HistogramGPU> keyframes;
	vector<int> framesClass;
	vector<vector<float> > clusters;
	int k;
};


#pragma once

#include "HistogramCPU.h"

class ClusteringCPU {
public:
	ClusteringCPU(vector<HistogramCPU> hists): features(hists){};

	void kmeans();
	void findKeyframes();
	void removeSimilarKeyframes();

	vector<HistogramCPU> getKeyframes() const { return keyframes; }

private:
	void estimateK();
	float euclidianDist(HistogramCPU h1, HistogramCPU h2);
	float euclidianDist(HistogramCPU h1, vector<float> cluster);
	float euclidianDist(vector<float> h1, vector<float> h2);

	int findNearestCluster(HistogramCPU hist);

	vector<HistogramCPU> features;
	vector<HistogramCPU> keyframes;
	vector<int> framesClass;
	vector<vector<float> > clusters;
	int k;
};


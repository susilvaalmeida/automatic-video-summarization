#pragma once

#include "HistogramMULT.h"

class ClusteringMULT {
public:
	ClusteringMULT(vector<HistogramMULT> hists): featuresM(hists){};

	void kmeans();
	void kmeansOMP();
	void findKeyframes();
	void removeSimilarKeyframes();

	vector<HistogramMULT> getKeyframes() const { return keyframes; }

private:
	void estimateK();
	float euclidianDist(HistogramMULT h1, HistogramMULT h2);
	float euclidianDist(HistogramMULT h1, vector<float> cluster);
	float euclidianDist(vector<float> h1, vector<float> h2);

	int findNearestCluster(HistogramMULT hist);
	int findNearestCluster(HistogramMULT hist, vector<vector<float> > clusters);

	void findClusterTH(int start, int end, int &delta, vector<vector<float> >& newClusters, vector<int>& newClusterSize);

	vector<HistogramMULT> featuresM;
	vector<HistogramMULT> keyframes;
	vector<int> framesClass;
	vector<vector<float> > clusters;
	int k;
};


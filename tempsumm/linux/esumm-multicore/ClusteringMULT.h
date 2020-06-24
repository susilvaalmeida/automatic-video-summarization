#pragma once

#include <vector>
using namespace std;

class ClusteringMULT
{
public:

	//for XMeans clustering
	ClusteringMULT();
	void callXmeans(string id, string parameterK = "1", string maxLeafSize = "5", string minBoxWidth = "0.05",
		string cutFactor = "0.5", string maxIterations = "300", string numSplits = "3", string maxCenters = "10"); //test parameters
	vector<int> getClosestToCentroids();

	vector<vector<int>> dataXmeans; 
	vector<vector<float>> centersXmeans;

	void setDataXmeans(vector<vector<int>> data)
	{
		this->dataXmeans = data;
	}

	 // for LBG clustering
	ClusteringMULT(string featDir);
	void changeEqualCols(vector<vector<int>>& features);

	//bag of visual words
	void callLBG(int qntCenters, string type);
	float distortionLBG(vector<float>& centeredData);
	float distortionNewLBG(vector<vector<float>>& centeredData, vector<int>& index);
	void meanInd(vector<vector<float>>& centeredData, vector<int>& index, int lenCenters);
	int findNearestCluster(vector<float>&, vector<vector<float>>&, int);
	void findClusterTH(int start, int end, vector<vector<float>>& clusters, int size, vector<int>& indexes);
	void findCodeWords(int start, int end, vector<vector<float>>& data, vector<vector<float>>& clusters, vector<int>& indexes);

	//histogram of visual words
	void codeWordsHistogram(int bins, string featDir, string type);

	//clustering of visual words vectors
	vector<int> clusterCodeWordsVector(int nscenes);

	vector<vector<float>> dataLBG; 
	vector<vector<float>> centersLBG;
	vector<vector<int>> histogramVisualWords;
};


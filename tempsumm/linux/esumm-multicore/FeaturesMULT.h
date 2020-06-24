#pragma once
#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class FeaturesMULT
{
public:
	FeaturesMULT(string descript, string frameDir, string featDir, vector<vector<int>>& shots, vector<vector<int>>& histograms, string type);
	Mat calcHog(Mat img);
	Mat calcSift(Mat img);
	Mat calcSurf(Mat img);

	void threadsFeatures(vector<vector<int>>& shots, vector<vector<int>>& histograms, int shotInicial, int shotFinal, int thrId);

	static void threadFilter(vector<string>& files, int idInicio, int idFim, string dir);
	static void filterFrames(string dir, string type);
	static void filterFrames(string dir);
	
	vector<string> frames;
	vector<vector<int>> representativeFrames;

	string framesDir, featDir, descript;
};



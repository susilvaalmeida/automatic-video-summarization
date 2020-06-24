#pragma once
#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class FeaturesGPU
{
public:
	FeaturesGPU(string descript, string frameDir, string featDir, vector<vector<int> >& shots, vector<vector<int> >& histograms);
	Mat calcHog(Mat img);
	Mat calcSift(Mat img);
	Mat calcSurf(Mat img);

	static void filterFrames(string dir);
	
	vector<vector<int> > representativeFrames;
};



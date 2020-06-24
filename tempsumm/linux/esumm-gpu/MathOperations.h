#pragma once
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <Eigen/Core>

using namespace std;
using namespace cv;

class MathOperations
{
public:
	static vector<double> conv(vector<double>& mA, vector<double>& mB, string shape);
	static vector<double> medianfilter(vector<double>& vetor, int W);
	
	static vector<float> dist(vector<vector<int> >& data, vector<float>& centers);
	static vector<float> dist(vector<vector<float> >& data, vector<float>& centers);
	static void dist(vector<vector<float> >& data, vector<float>& centers, vector<float>& dist);
	static float dist(vector<float>& data, vector<float>& centers);

	static vector<double> diff(const vector<double>& vetor);
	static Eigen::VectorXd diff(Eigen::VectorXd vetor);

	static void meanCols(vector<vector<float> >& data, vector<float>&);

	static vector<float> histHSV(Mat frame, int bins);
	static vector<float> histRGB(Mat frame, int bins);
	static float variance(Mat frame);
	static float correlation(vector<float>& img1, vector<float>& img2);

};


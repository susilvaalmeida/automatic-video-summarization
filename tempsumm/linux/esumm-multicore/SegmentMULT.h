#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <vector>
#include <Eigen/Sparse>
#include "FileOperations.h"

using namespace std;
using namespace cv;

class SegmentMULT
{
public:
	SegmentMULT();
	SegmentMULT(string framesDir, string type);

	void threadSegment(int idInicial, int idFinal);

	void frameHistRGB(Mat frame, int bins, int indice = -1);
	void frameVariance(Mat grayframe, int indice = -1);
	void cosineDissimilarity(vector<int> hist1, vector<int> hist2, int indice = -1);
	vector<double> filterDissimilarity(int qntNeighbors);
	vector<double> findEffects();
	vector<vector<int>> dissolveDetectionIndexes();
	vector<vector<int>> mergeIntervals(vector<vector<int>> intervals, int dist);
	vector<double> smoothSpline(vector<double> mvar, double parameter);
	Eigen::SparseMatrix<double> createSparse(vector<vector<int>> mat, vector<int> diags, int rows, int cols);	
	Eigen::SparseMatrix<double> createSparse(vector<int> mat, int diags, int rows, int cols);	

	vector<double> variances;
	vector<double> dissimilarity;

	vector<vector<int>> shots;
	vector<vector<int>> histograms;
	int nscenes;

	vector<string> frames;
};


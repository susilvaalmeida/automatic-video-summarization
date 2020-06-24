#include "FileOperations.h"
#include "FeaturesMULT.h"
#include "ClusteringMULT.h"
#include "MathOperationsMULT.h"

#include <algorithm>
#include <numeric>
#include <iomanip>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>


#include <thread>
#include <omp.h>


#include "Results.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

FeaturesMULT::FeaturesMULT(string descript, string framesDir, string featDir, vector<vector<int>>& shots, vector<vector<int>>& histograms, string type)
{
	this->descript = descript;
	this->framesDir = framesDir;
	this->featDir = featDir;

	frames = FileOperations::listFiles(framesDir, ".jpg");
	//sort(frames.begin(), frames.end());

	representativeFrames.resize(shots.size());

	int max_num_threads = omp_get_max_threads();
	int div = shots.size() / max_num_threads;
	int idInicial = 0;
	int idFinal = div;

	if(type == "thr")
	{
		vector<thread> vThreads;
		for(int i=0; i<max_num_threads; i++)
		{
			if(i==max_num_threads-1)
				vThreads.push_back(thread(&FeaturesMULT::threadsFeatures, this, std::ref(shots), std::ref(histograms), idInicial, shots.size(),i));
			else
			{
				vThreads.push_back(thread(&FeaturesMULT::threadsFeatures, this, std::ref(shots), std::ref(histograms), idInicial, idFinal,i));
				idInicial = idFinal;
				idFinal = idInicial + div;
			}
		}
		for(int i=0; i<max_num_threads; i++)
		{
			vThreads[i].join();
		}
	}

	else if(type == "omp")
	{
#pragma omp parallel for
		for(int i=0; i<max_num_threads; i++)
		{
			if(i==max_num_threads-1)
				threadsFeatures(shots, histograms, idInicial, (int)shots.size(),i);
			else
			{
				threadsFeatures(shots, histograms, idInicial, idFinal,i);
				idInicial = idFinal;
				idFinal = idInicial + div;
			}
		}

	}
}

void FeaturesMULT::threadsFeatures(vector<vector<int>>& shots, vector<vector<int>>& histograms, int shotInicial, int shotFinal, int threadId)
{
	vector<vector<int>> features;
	ClusteringMULT clusterHists;

	for(int i=shotInicial; i<shotFinal; i++)
	{
		features.clear();
		vector<vector<float>> centers;

		for(int j=shots[i][0]; j<=shots[i][1]; j++) //pega os features de cada shot encontrado
			features.push_back(histograms[j]);

		stringstream ss;
		ss << threadId;

		clusterHists.setDataXmeans(features);
		clusterHists.callXmeans(ss.str());
		vector<int> ind = clusterHists.getClosestToCentroids();

		int iniIndex = shots[i][0];
		for(int j=0; j<(int)ind.size(); j++)
		{
			ind[j] = ind[j] + iniIndex - 1;
		}
		representativeFrames[i] = ind;

		for(int j=0; j<(int)representativeFrames[i].size(); j++)
		{
			if(representativeFrames[i][j] > (int)frames.size() || representativeFrames[i][j] < 0)
				continue;

			Mat imgc = imread(frames[representativeFrames[i][j]], CV_LOAD_IMAGE_GRAYSCALE);

			Mat mat_features;
			if(descript == "hog")
				mat_features = calcHog(imgc);
			else if(descript == "sift")
				mat_features = calcSift(imgc);
			else if(descript == "surf")
				mat_features = calcSurf(imgc);

			vector<vector<float>> featVec(mat_features.rows);
			for(int a=0; a<mat_features.rows; a++)
			{
				vector<float> tmp(mat_features.cols);
				for(int b=0; b<mat_features.cols; b++)
					tmp[b] = mat_features.at<float>(a,b);
				featVec[a] = tmp;
			}

			//salve feature file
			//FeatureFile file; file.setFeature(featVec);
			string name = featDir + FileOperations::getSimpleName(frames[representativeFrames[i][j]]) + ".feat";
			string content;
			for(int kk=0; kk<(int)featVec.size(); kk++)
			{
				for(int kkk=0; kkk<(int)featVec[kk].size(); kkk++)
				{
					std::ostringstream o;
					o << std::fixed;
					o << std::setprecision(6) << featVec[kk][kkk];
					content += o.str() + " ";
				}
				content += "\n";
			}
			FileOperations::createFile(name, content);

			//cout << name << endl;
			//copie frame to features directory
			FileOperations::copyFile(frames[representativeFrames[i][j]], featDir);
		}
	}
}

Mat FeaturesMULT::calcHog(Mat img)
{
	cv::HOGDescriptor hogMULT(Size(16,16), Size(16,16), Size(8,8), Size(8,8), 9);
	vector<float> descriptors;
	vector<Point> locs;

	hogMULT.compute(img, descriptors, Size(8,8), Size(), locs);


	Mat hogMat(descriptors);
	cv::transpose(hogMat.clone().reshape(0,9), hogMat);

	return hogMat;
}

Mat FeaturesMULT::calcSift(Mat img)
{
	Mat siftMat;

	vector<KeyPoint> keypoints;

	SIFT sift;
	sift(img,Mat(),keypoints,siftMat);

	return siftMat;
}

Mat FeaturesMULT::calcSurf(Mat img)
{
	Mat surfMat;

	vector<KeyPoint> keypoints;

	SURF surf;
	surf(img,Mat(),keypoints,surfMat);

	return surfMat;
}

void FeaturesMULT::threadFilter(vector<string>& files, int idInicio, int idFim, string dir)
{
	for(int i=idInicio; i<idFim-1; i++)
	{
		cv::Mat imPivot = imread(files[i]);
		cv::Mat nextImg = imread(files[i+1]);

		vector<float> himPivot = MathOperationsMULT::histHSV(imPivot, 16);
		for(int j=0; j<(int)himPivot.size(); j++)
			himPivot[j] /= imPivot.rows*imPivot.cols;

		vector<float> himNext = MathOperationsMULT::histHSV(nextImg, 16);
		for(int j=0; j<(int)himNext.size(); j++)
			himNext[j] /= nextImg.rows*nextImg.cols;

		vector<float> sub;
		for(int j=0; j<(int)himPivot.size(); j++)
			sub.push_back(abs(himNext[j] - himPivot[j]));


		float manhattandis = (float)std::accumulate(sub.begin(), sub.end(), 0.0);

		vector<float> him1 = MathOperationsMULT::histRGB(imPivot, 256);
		for(int j=0; j<(int)him1.size(); j++)
			him1[j] /= imPivot.rows*imPivot.cols;

		vector<float> him2 = MathOperationsMULT::histRGB(nextImg, 256);
		for(int j=0; j<(int)him2.size(); j++)
			him2[j] /= nextImg.rows*nextImg.cols;

		float corrHisto = MathOperationsMULT::correlation(him1, him2);

		cv::Mat im1, im2;
		cvtColor(imPivot, im1, CV_BGR2GRAY);
		cvtColor(nextImg, im2, CV_BGR2GRAY);

		vector<float> im1V, im2V;
		for(int k=0; k<im1.rows; k++)
		{
			for(int j=0; j<im1.cols; j++)
				im1V.push_back((float)im1.at<uchar>(k,j));
		}

		for(int k=0; k<im2.rows; k++)
		{
			for(int j=0; j<im2.cols; j++)
				im2V.push_back((float)im2.at<uchar>(k,j));
		}

		float corrIma = MathOperationsMULT::correlation(im1V, im2V);

		//cout << files[i] << " " << files[i+1] << endl;
		cout << manhattandis << " " << corrHisto << " " << corrIma << endl;
		if(manhattandis < 0.5 && corrHisto > 0.5 && corrIma > 0.5)
		{
			cout << "removed " << files[i] << endl;
			FileOperations::deleteFile(files[i]);
		}
	}
}

void FeaturesMULT::filterFrames(string dir, string type)
{
	double th = 0.75;
	double thvar = 700;

	vector<string> framesSummary =  FileOperations::listFiles(dir, ".jpg");
	//sort(framesSummary.begin(), framesSummary.end());

	int max_num_threads = omp_get_max_threads();


	int div = (int)framesSummary.size() / max_num_threads;
	int idInicial = 0;
	int idFinal = div;


	if(type == "thr")
	{
		vector<thread> vThreads;

		for(int i=0; i<max_num_threads; i++)
		{
			if(i==max_num_threads-1)
				vThreads.push_back(thread(&FeaturesMULT::threadFilter, std::ref(framesSummary), idInicial, framesSummary.size(), dir));
			else
			{
				vThreads.push_back(thread(&FeaturesMULT::threadFilter, std::ref(framesSummary), idInicial, idFinal, dir));
				idInicial = idFinal;
				idFinal = idInicial + div;
			}
		}

		for(int i=0; i<max_num_threads; i++)
			vThreads[i].join();
	}
	else if(type == "omp")
	{
#pragma omp parallel for
		for(int i=0; i<max_num_threads; i++)
		{
			if(i==max_num_threads-1)
				threadFilter(framesSummary,idInicial,(int)framesSummary.size(), dir);
			else
			{
				threadFilter(framesSummary,idInicial,idFinal, dir);
				idInicial = idFinal;
				idFinal = idInicial + div;
			}
		}
	}


	framesSummary =  FileOperations::listFiles(dir, ".jpg");
	//sort(framesSummary.begin(), framesSummary.end());

	for(int i=0; i<(int)framesSummary.size(); i++)
	{
		cv::Mat imPivot = imread(framesSummary[i],CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat im1;
		//cvtColor(imPivot, im1, CV_BGR2GRAY);

		float var = MathOperationsMULT::variance(im1);

		//cout << "var: " << var << " thvar: " << thvar << endl;
		if(var <= thvar)
			FileOperations::deleteFile(framesSummary[i]);
	}
}

void FeaturesMULT::filterFrames(string dir)
{
	//double th = 0.75;
	double thvar = 700;

	vector<string> framesSummary =  FileOperations::listFiles(dir, ".jpg");
	//sort(framesSummary.begin(), framesSummary.end());

	for(int i=0; i<(int)framesSummary.size()-1; i++)
	{
		cv::Mat imPivot = imread(framesSummary[i]);
		cv::Mat nextImg = imread(framesSummary[i+1]);

		if(imPivot.channels() < 3 || nextImg.channels() < 3)
			continue;

		vector<float> himPivot = MathOperationsMULT::histHSV(imPivot, 16);
		for(int j=0; j<(int)himPivot.size(); j++)
			himPivot[j] /= imPivot.rows*imPivot.cols;

		vector<float> himNext = MathOperationsMULT::histHSV(nextImg, 16);
		for(int j=0; j<(int)himNext.size(); j++)
			himNext[j] /= nextImg.rows*nextImg.cols;

		vector<float> sub;
		for(int j=0; j<(int)himPivot.size(); j++)
			sub.push_back(abs(himNext[j] - himPivot[j]));


		float manhattandis = (float)std::accumulate(sub.begin(), sub.end(), 0.0);

		vector<float> him1 = MathOperationsMULT::histRGB(imPivot, 256);
		for(int j=0; j<(int)him1.size(); j++)
			him1[j] /= imPivot.rows*imPivot.cols;

		vector<float> him2 = MathOperationsMULT::histRGB(nextImg, 256);
		for(int j=0; j<(int)him2.size(); j++)
			him2[j] /= nextImg.rows*nextImg.cols;

		float corrHisto = MathOperationsMULT::correlation(him1, him2);

		cv::Mat im1, im2;
		cvtColor(imPivot, im1, CV_BGR2GRAY);
		cvtColor(nextImg, im2, CV_BGR2GRAY);

		vector<float> im1V, im2V;
		for(int k=0; k<im1.rows; k++)
		{
			for(int j=0; j<im1.cols; j++)
				im1V.push_back((float)im1.at<uchar>(k,j));
		}

		for(int k=0; k<im2.rows; k++)
		{
			for(int j=0; j<im2.cols; j++)
				im2V.push_back((float)im2.at<uchar>(k,j));
		}

		float corrIma = MathOperationsMULT::correlation(im1V, im2V);

		//cout << framesSummary[i] << " " << framesSummary[i+1] << endl;
		//cout << manhattandis << " " << corrHisto << " " << corrIma << endl;
		if(manhattandis < 0.5 && corrHisto > 0.5 && corrIma > 0.5)
		{
			//cout << "removing " << framesSummary[i] << endl;
			FileOperations::deleteFile(framesSummary[i]);
		}
	}

	framesSummary =  FileOperations::listFiles(dir, ".jpg");

	for(int i=0; i<(int)framesSummary.size(); i++)
	{
		cv::Mat imPivot = imread(framesSummary[i]);
		cv::Mat im1;
		cvtColor(imPivot, im1, CV_BGR2GRAY);

		float var = MathOperationsMULT::variance(im1);

		//cout << framesSummary[i] << " -> " << var << " " << thvar << endl;
		if(var <= thvar)
			FileOperations::deleteFile(framesSummary[i]);
	}
}

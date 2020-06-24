#include "FileOperations.h"
#include "FeaturesGPU.h"
#include "ClusteringGPU.h"
#include "MathOperations.h"

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
#include <opencv2/nonfree/gpu.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "Results.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

FeaturesGPU::FeaturesGPU(string descript, string framesDir, string featDir, vector<vector<int> >& shots, vector<vector<int> >& histograms)
{
	cv::TickMeter timeLocal;
		double time = 0.0;

	vector<string> frames = FileOperations::listFiles(framesDir, ".jpg");
	//sort(frames.begin(), frames.end());

	vector<vector<int> > features;
	ClusteringGPU clusterHists;

	for(int i=0; i<(int)shots.size(); i++)
	{
		features.clear();
		vector<vector<float> > centers;

		for(int j=shots[i][0]; j<=shots[i][1]; j++) //pega os features de cada shot encontrado
			features.push_back(histograms[j]);

		clusterHists.setDataXmeans(features);
		clusterHists.callXmeans();
		vector<int> ind = clusterHists.getClosestToCentroids();

		int iniIndex = shots[i][0];
		for(int j=0; j<(int)ind.size(); j++)
		{
			ind[j] = ind[j] + iniIndex - 1;
		}

		representativeFrames.push_back(ind);

		for(int j=0; j<(int)representativeFrames[i].size(); j++)
		{
			if(representativeFrames[i][j] > (int)frames.size() || representativeFrames[i][j] < 0)
				continue;

			Mat imgc = imread(frames[representativeFrames[i][j]], CV_LOAD_IMAGE_GRAYSCALE);

			timeLocal.reset(); timeLocal.start();

			Mat mat_features;
			if(descript == "hog")
				mat_features = calcHog(imgc);
			else if(descript == "sift")
				mat_features = calcSift(imgc);
			else if(descript == "surf")
				mat_features = calcSurf(imgc);

			timeLocal.stop(); time += timeLocal.getTimeSec();

			vector<vector<float> > featVec(mat_features.rows);
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

			//copie frame to features directory
			FileOperations::copyFile(frames[representativeFrames[i][j]], featDir);
		}
	}
	Results *result;
	result = Results::getInstance();
	result->setFeatDetecDescParallelPart(time);
}

Mat FeaturesGPU::calcHog(Mat img)
{
	cv::TickMeter timeLocal;
	double time = 0.0;

	cv::gpu::GpuMat imgGPU;
	imgGPU.upload(img);

	timeLocal.reset(); timeLocal.start();
	cv::gpu::HOGDescriptor hog(cv::Size(16,16));
	GpuMat descriptors;
	hog.getDescriptors(imgGPU, cv::Size(8,8), descriptors);
	timeLocal.stop(); time += timeLocal.getTimeSec();

	imgGPU.release();

	Mat hogMat;
	descriptors.download(hogMat);

	timeLocal.reset(); timeLocal.start();
	cv::transpose(hogMat.clone().reshape(0,9), hogMat);
	timeLocal.stop(); time += timeLocal.getTimeSec();

	Results *result;
	result = Results::getInstance();

	double total = result->getFeatDetecDescWithoutGpuCopy();
	if(total == -1)
		total = 0.0;
	result->setFeatDetecDescWithoutGpuCopy(total+time);

	return hogMat;
}

Mat FeaturesGPU::calcSift(Mat img)
{
	Mat siftMat;

	vector<KeyPoint> keypoints;

	SIFT sift;
	sift(img,Mat(),keypoints,siftMat);

	return siftMat;
}

Mat FeaturesGPU::calcSurf(Mat img)
{
	cv::gpu::GpuMat imgGPU;
	imgGPU.upload(img);

	cv::gpu::GpuMat descriptors, keypoints;

	cv::gpu::SURF_GPU surf;
	surf(imgGPU,GpuMat(),keypoints,descriptors);

	Mat surfMat;
	descriptors.download(surfMat);

	return surfMat;
}

void FeaturesGPU::filterFrames(string dir)
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

		vector<float> himPivot = MathOperations::histHSV(imPivot, 16);
		for(int j=0; j<(int)himPivot.size(); j++)
			himPivot[j] /= imPivot.rows*imPivot.cols;

		vector<float> himNext = MathOperations::histHSV(nextImg, 16);
		for(int j=0; j<(int)himNext.size(); j++)
			himNext[j] /= nextImg.rows*nextImg.cols;

		vector<float> sub;
		for(int j=0; j<(int)himPivot.size(); j++)
			sub.push_back(abs(himNext[j] - himPivot[j]));


		float manhattandis = std::accumulate(sub.begin(), sub.end(), 0.0);

		vector<float> him1 = MathOperations::histRGB(imPivot, 256);
		for(int j=0; j<(int)him1.size(); j++)
			him1[j] /= imPivot.rows*imPivot.cols;

		vector<float> him2 = MathOperations::histRGB(nextImg, 256);
		for(int j=0; j<(int)him2.size(); j++)
			him2[j] /= nextImg.rows*nextImg.cols;

		float corrHisto = MathOperations::correlation(him1, him2);

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

		float corrIma = MathOperations::correlation(im1V, im2V);

		//cout << framesSummary[i] << " " << framesSummary[i+1] << endl;
		//cout << manhattandis << " " << corrHisto << " " << corrIma << endl;
		if(manhattandis < 0.5 && corrHisto > 0.5 && corrIma > 0.5)
		{
			//cout << "removing " << framesSummary[i] << endl;
			FileOperations::deleteFile(framesSummary[i]);
		}
	}

	framesSummary =  FileOperations::listFiles(dir, ".jpg");
	//sort(framesSummary.begin(), framesSummary.end());

	for(int i=0; i<(int)framesSummary.size(); i++)
	{
		cv::Mat imPivot = imread(framesSummary[i]);
		cv::Mat im1;
		cvtColor(imPivot, im1, CV_BGR2GRAY);

		float var = MathOperations::variance(im1);

		//cout << framesSummary[i] << " -> " << var << " " << thvar << endl;
		if(var <= thvar)
			FileOperations::deleteFile(framesSummary[i]);
	}
}

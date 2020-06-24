#include "MathOperationsMULT.h"
#include <numeric>
#include <math.h>
#include <iostream>

vector<double> MathOperationsMULT::conv(vector<double>& mA, vector<double>& mB, string shape){

	vector<double> result(mA.size() + mB.size() - 1);

	for(int i=0; i<(int)result.size(); i++)
		result[i] = 0;

	for(int i=0;i<(int)result.size();i++){
		int i1 = i;
		double tmp = 0.0;
		for(int j=0;j<(int)mB.size();j++){
			if(i1>=0 && i1<(int)mA.size()){
				tmp = tmp + mA[i1]*mB[j];
			}
			i1--;
			result[i] = tmp;
		}
	}
	if(shape == "same"){
		if(mB.size() == 2){
			result.erase(result.begin()); //apaga primeiro elemento
		}
		else if(mB.size() == 3){
			result.erase(result.begin()); //apaga primeiro e ultimo elementos
			result.erase(result.end()-1);
		}
	}

	return result;
}

vector<double> MathOperationsMULT::medianfilter(vector<double>& vetor, int W){
	vector<double> result(vetor.size());
	double *window = (double*)malloc(W*sizeof(double));
	for(int i=2; i<(int)vetor.size()-2; ++i){
		for(int j=0; j<W; ++j)
			window[j] = vetor[i-2+j];
		for(int j=0; j<3; ++j){
			int min = j;
			for(int k=j+1; k<W; ++k){
				if(window[k] < window[min])
					min = k;
			}
			double temp = window[j];
			window[j] = window[min];
			window[min] = temp;
		}
		result[i-2] = window[2];
	}
	free(window);
	return result;
}

float MathOperationsMULT::dist(vector<float>& data, vector<float>& centers)
{
	float soma = 0.0;
	for(int i=0; i<(int)data.size(); i++)
	{
		soma += pow(data[i] - centers[i], 2);
	}
	return sqrt(soma);
}

vector<float> MathOperationsMULT::dist(vector<vector<int>>& data, vector<float>& centers)
{
	vector<float> distances;
	for(int i=0; i<(int)data.size(); i++)
	{
		float soma = 0.0;
		for(int j=0; j<(int)data[i].size(); j++)
		{
			soma += pow((float)data[i][j] - centers[j], 2);
		}
		distances.push_back(sqrt(soma));
	}
	return distances;
}

vector<float> MathOperationsMULT::dist(vector<vector<float>>& data, vector<float>& centers)
{
	vector<float> distances;
	for(int i=0; i<(int)data.size(); i++)
	{
		float soma = 0;
		for(int j=0; j<(int)data[i].size(); j++)
			soma += pow(data[i][j] - centers[j], 2);
		distances.push_back(sqrt(soma));
	}
	return distances;
}

void MathOperationsMULT::dist(vector<vector<float>>& data, vector<float>& centers, vector<float>& dist)
{
	for(int i=0; i<(int)data.size(); i++)
	{
		float soma = 0;
		for(int j=0; j<(int)data[0].size(); j++)
			soma += pow(data[i][j] - centers[j], 2);

		if((int)dist.size() > i)
			dist[i] = soma;
		else
			dist.push_back(soma);
	}
}

vector<double> MathOperationsMULT::diff(const vector<double>& in)
{
	vector<double> res(in.size()-1);
	for(int i=1, j=0; i < (int)in.size(); i++,j++)
		res[j] =  (in[i]) - (in[i-1]) ;
	return res;
}

Eigen::VectorXd MathOperationsMULT::diff(Eigen::VectorXd vetor)
{
	Eigen::VectorXd res(vetor.size()-1);
	for(int i=1, j=0; i<vetor.size(); i++,j++)
		res(j) = vetor(i) - vetor(i-1);
	return res;
}

void MathOperationsMULT::meanCols(vector<vector<float>>& data, vector<float>& means)
{
	for(int i=0; i<(int)data[0].size(); i++)
	{
		float sum = 0;
		for(int j=0; j<(int)data.size(); j++)
			sum += data[j][i];
		
		if((int)means.size() > i)
			means[i] = sum / data.size();
		else
			means.push_back(sum / data.size());	
	}
}

vector<float> MathOperationsMULT::histHSV(Mat frame, int bins)
{
	cv::Mat hsvImg;
	cvtColor(frame, hsvImg, CV_RGB2HSV);

	std::vector<cv::Mat> hsvChannels(3);
	cv::split(hsvImg, hsvChannels);

	vector<float> histogram(bins);
	for(int i=0; i<hsvChannels[0].rows; i++)
	{
		for(int j=0; j<hsvChannels[0].cols; j++)
			histogram[hsvChannels[0].at<uchar>(i,j) / bins]++;
	}

	return histogram;
}

vector<float> MathOperationsMULT::histRGB(Mat frame, int bins)
{
	vector<float> hist(bins*3);
	for(int i=0; i<frame.rows; i++){
		for(int j=0; j<frame.cols; j++){
			hist[frame.at<cv::Vec3b>(i,j)[0]+2*bins]++;
			hist[frame.at<cv::Vec3b>(i,j)[1]+bins]++;
			hist[frame.at<cv::Vec3b>(i,j)[2]]++;
		}
	}
	return hist;
}

float MathOperationsMULT::variance(Mat grayframe)
{
	if(grayframe.channels() == 1)
	{
		Scalar mean, stdDev;
		meanStdDev(grayframe, mean, stdDev);
		float variance = pow(stdDev[0],2);
		return variance;
	}
	return -1;
}

float MathOperationsMULT::correlation(vector<float>& img1, vector<float>& img2)
{
	float m1 = (float)std::accumulate(img1.begin(), img1.end(),0.0) / img1.size();
	float m2 = (float)std::accumulate(img2.begin(), img2.end(),0.0) / img2.size();

	vector<float> img12, img22;

	for(int i=0; i<(int)img1.size(); i++)
	{
		img12.push_back(img1[i] - m1);
		img22.push_back(img2[i] - m2);
	}

	vector<float> mult;
	for(int i=0; i<(int)img12.size(); i++)
		mult.push_back(img12[i] * img22[i]);

	float a = std::accumulate(mult.begin(), mult.end(), 0.0);

	vector<float> exp1, exp2;
	for(int i=0; i<(int)img12.size(); i++)
	{
		exp1.push_back(pow(img12[i],2));
		exp2.push_back(pow(img22[i],2));
	}

	float b = sqrt(std::accumulate(exp1.begin(), exp1.end(),0.0)) * 
		sqrt(std::accumulate(exp2.begin(), exp2.end(),0.0));

	return a/b;
}

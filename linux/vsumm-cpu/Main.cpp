//============================================================================
// Name        : vsumm-cpu.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <iomanip>

#include "Defines.h"
#include "FileOperations.h"
#include "DecoderCPU.h"
#include "HistogramCPU.h"
#include "ClusteringCPU.h"
#include "Results.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <string>


using namespace std;


int main(int argc, char **argv) {

	Results *result;
	result = Results::getInstance();

	cv::TickMeter timeGlobal, timeLocal;

	string video_path(argv[1]);
	vector<string> videoNames = FileOperations::listFiles(video_path, ".avi");
	sort(videoNames.begin(), videoNames.end());


	for(int i=0; i<(int)videoNames.size(); i++)
	{
		cout << "Processing video " << videoNames[i] << "... ";
		//-----------------------CREATE DIRECTORIES------------------------------//
		string frameDir = "frames/";
		string summaryDir = "summary-"+FileOperations::getSimpleName(videoNames[i])+"/";

		if(FileOperations::createDir(frameDir) == 0)
		{
			FileOperations::deleteDir(frameDir);
			FileOperations::createDir(frameDir);
		}
		if(FileOperations::createDir(summaryDir) == 0)
		{
			vector<string> frame = FileOperations::listFiles(summaryDir, ".jpg");
			if(frame.size() > 0)
				continue;
		}
		//-----------------------------------------------------------------------//


		//-----------------------GET VIDEO INFORMATION---------------------------//
		result->setArch("cpu");
		result->setVideoName(FileOperations::getSimpleName(videoNames[i]));

		vector<string> splitStr = FileOperations::split(FileOperations::getSimpleName(videoNames[i]), '_');

		result->setLength(atoi(splitStr[1].c_str()));
		if(atoi(splitStr[1].c_str()) == 30)
			result->setResolution(splitStr[2]);
		else
			result->setResolution(splitStr[3]);
		//-----------------------------------------------------------------------//



		timeGlobal.reset(); timeGlobal.start();
		timeLocal.reset(); timeLocal.start();
		//------------------------VIDEO SEGMENTATION-----------------------------//
		DecoderCPU::saveFrames(videoNames[i], frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());



		timeLocal.reset(); timeLocal.start();
		//------------------------FEATURE EXTRACTION-----------------------------//
		vector<string> frameNames = FileOperations::listFiles(frameDir, ".jpg");
		sort(frameNames.begin(), frameNames.end());

		vector<HistogramCPU> frameHistograms = FeaturesCPU::computeAllHist(frameNames);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setFeatExtraction(timeLocal.getTimeSec());





		timeLocal.reset(); timeLocal.start();
		//------------------------FRAMES CLUSTERING------------------------------//
		ClusteringCPU clust(frameHistograms);
		clust.kmeans();
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setClustering(timeLocal.getTimeSec());



		timeLocal.reset(); timeLocal.start();
		//------------------------KEYFRAME EXTRACTION----------------------------//
		clust.findKeyframes();
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setKeyframeExtraction(timeLocal.getTimeSec());



		timeLocal.reset(); timeLocal.start();
		//-------------------ELIMINATION OF SIMILAR KEYFRAMES--------------------//
		clust.removeSimilarKeyframes();
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setEliminateSimilar(timeLocal.getTimeSec());




		//---------------------GENERATE SUMMARY----------------------------------//
		vector<HistogramCPU> finalKeyframes = clust.getKeyframes();
		for(int j=0; j<(int)finalKeyframes.size(); j++)
		{
			if(finalKeyframes[j].getIdFrame() >= frameNames.size())
				continue;
			FileOperations::copyFile(frameNames[finalKeyframes[j].getIdFrame()], summaryDir);
		}
		FileOperations::deleteDir(frameDir);
		//-----------------------------------------------------------------------//


		timeGlobal.stop();
		result->setTotal(timeGlobal.getTimeSec());

		result->print();
		result->save();
		cout << endl;

	}
	return 0;
}

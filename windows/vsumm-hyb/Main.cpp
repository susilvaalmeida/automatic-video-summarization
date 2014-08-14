
#include <iostream>
#include <vector>
#include <iomanip>


#include "Defines.h"
#include "FileOperations.h"
#include "DecoderMULT.h" 
#include "HistogramGPU.cuh"
#include "ClusteringGPU.cuh"
#include "Results.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>


#include <string>

using namespace std;

extern "C"
vector<HistogramGPU> callComputeHist(string frameDir, double &time);
int main(int argc, char **argv) {

	Results *result;
	result = Results::getInstance();

	cv::TickMeter timeGlobal, timeLocal;

	string video_path(argv[1]);
	vector<string> videoNames = FileOperations::listFiles(video_path, ".avi");
	sort(videoNames.begin(), videoNames.end());

	for(int i=0; i<videoNames.size(); i++)
	{
		cout << "Processing video " << videoNames[i] << "..." << endl;

		//-----------------------CREATE DIRECTORIES------------------------------//
		string frameDir = "framesHYB-"+FileOperations::getSimpleName(videoNames[i])+"\\";
		string summaryDir = "summaryHYB-"+FileOperations::getSimpleName(videoNames[i])+"\\";

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
		result->setArch("hyb");
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
		//------------------------VIDEO SEGMENTATION-----------------------------/	
		DecoderMULT::saveFrames(videoNames[i], frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());


		vector<string> frameNames;
		frameNames = FileOperations::listFiles(frameDir, ".jpg");
		sort(frameNames.begin(), frameNames.end());


		double d;
		//------------------------FEATURE EXTRACTION-----------------------------//
		vector<HistogramGPU> frameHistograms = callComputeHist(frameDir, d);

		//-----------------------------------------------------------------------//

		



		timeLocal.reset(); timeLocal.start();
		//------------------------FRAMES CLUSTERING------------------------------//
		ClusteringGPU clust(frameHistograms);
		clust.kmeansGPU();
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
		vector<HistogramGPU> finalKeyframes = clust.getKeyframes();
		for(int ki=0; ki<(int)finalKeyframes.size(); ki++)
		{
			if(finalKeyframes[ki].getIdFrame() >= frameNames.size())
				continue;
			FileOperations::copyFile(frameNames[finalKeyframes[ki].getIdFrame()], summaryDir);
		}
		FileOperations::deleteDir(frameDir);
		//-----------------------------------------------------------------------//

		timeGlobal.stop();
		result->setTotal(timeGlobal.getTimeSec());


		cout << "Summary generated. Saving results..." << endl;
		cout << timeGlobal.getTimeSec() << endl;
		result->save();
		cout << endl;

		//cudaDeviceReset();
		//exit(0);
	}

	return 0;
}

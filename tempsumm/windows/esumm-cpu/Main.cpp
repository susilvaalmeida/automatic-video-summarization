#include <iostream>
#include <vector>
#include <iomanip>

#include "Defines.h"
#include "FileOperations.h"
#include "Results.h"

#include "DecoderCPU.h"
#include "SegmentCPU.h"
#include "FeaturesCPU.h"
#include "ClusteringCPU.h"

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
		string dname(argv[2]);
		
		cout << "Processing video " << videoNames[i] << "..." << endl;
		//-----------------------CREATE DIRECTORIES------------------------------//
		string frameDir = "framesCPU-"+dname+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";
		string featuresDir = "featuresCPU-"+dname+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";

		string summaryDir = "summaryCPU-"+dname+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";

		if(FileOperations::createDir(frameDir) == 0)
		{
			FileOperations::deleteDir(frameDir);
			FileOperations::createDir(frameDir);
		}
		if(FileOperations::createDir(featuresDir) == 0)
		{
			FileOperations::deleteDir(featuresDir);
			FileOperations::createDir(featuresDir);
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
		//------------------------VIDEO DECODING---------------------------------//
		DecoderCPU::saveFrames(videoNames[i], frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//------------------------VIDEO SEGMENTATION-----------------------------//
		SegmentCPU segCPU(frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setSegmentation(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//-----------------DETECTION OF REPRESENTATIVE FRAMES--------------------//
		result->setDescriptor(argv[2]);
		FeaturesCPU featCPU(argv[2], frameDir, featuresDir, segCPU.shots, segCPU.histograms);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setFeatDetecDesc(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//------------------------BAG OF VISUAL WORDS----------------------------//
		ClusteringCPU lbg(featuresDir);
		lbg.callLBG(150);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setLbgClustering(timeLocal.getTimeSec());





		timeLocal.reset(); timeLocal.start();
		//-------------------HISTOGRAM OF VISUAL WORDS---------------------------//
		lbg.codeWordsHistogram(150,featuresDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setHistogramWords(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//-------------------VISUAL WORD VECTORS CLUSTERING----------------------//
		vector<int> indexCenters = lbg.clusterCodeWordsVector(segCPU.nscenes);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setWordsClutering(timeLocal.getTimeSec());




		//---------------------GENERATE SUMMARY----------------------------------//
		vector<int> vetorRepresFrames;
		for(int k=0; k<(int)featCPU.representativeFrames.size(); k++)
		{
			for(int j=0; j<(int)featCPU.representativeFrames[k].size(); j++)
			{
				vetorRepresFrames.push_back(featCPU.representativeFrames[k][j]);
			}
		}

		vector<string> frames = FileOperations::listFiles(frameDir, ".jpg");
		sort(frames.begin(), frames.end());

		for(int j=0; j<(int)indexCenters.size(); j++)
		{
			if(indexCenters[j] >= vetorRepresFrames.size() || vetorRepresFrames[indexCenters[j]] >= frames.size())
				continue;
			FileOperations::copyFile(frames[vetorRepresFrames[indexCenters[j]]], summaryDir);
		}
		//-----------------------------------------------------------------------//


		timeLocal.reset(); timeLocal.start();
		//------------------------------FILTERING--------------------------------//
		FeaturesCPU::filterFrames(summaryDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setFilter(timeLocal.getTimeSec());


		result->setTotal(timeGlobal.getTimeSec());

		FileOperations::deleteDir(frameDir);
		FileOperations::deleteDir(featuresDir);

		cout << "Summary generated. Saving results..." << endl;
		cout << timeGlobal.getTimeSec() << endl;
		result->save();
		cout << endl;

	}

	return 0;
}

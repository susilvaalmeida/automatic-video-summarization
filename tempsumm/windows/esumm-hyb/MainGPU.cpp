#include <iostream>
#include <vector>
#include <iomanip>

#include "Defines.h"
#include "FileOperations.h"
#include "Results.h"

#include "DecoderGPU.h"
#include "SegmentGPU.h"
#include "FeaturesGPU.h"
#include "ClusteringGPU.h"

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
		cout << "Processing video " << videoNames[i] << "..." << endl;

		

		//-----------------------CREATE DIRECTORIES------------------------------//
		string dname(argv[2]);
		string dectype(argv[3]);

		string frameDir = "framesGPU-"+dname+"-dec"+dectype+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";
		string featuresDir = "featuresGPU-"+dname+"-dec"+dectype+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";
		string summaryDir = "summaryGPU-"+dname+"-dec"+dectype+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";

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
		result->setArch("gpu-"+dectype);
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
		if(dectype == "1")
			DecoderGPU::saveFrames(videoNames[i], frameDir);
		else if(dectype == "2")
			DecoderGPU::saveFramesGPU(videoNames[i], frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());
		result->setDecodingType(dectype);



		timeLocal.reset(); timeLocal.start();
		//------------------------VIDEO SEGMENTATION-----------------------------//
		SegmentGPU segGPU(frameDir);
		//-----------------------------------------------------------------------//

		timeLocal.stop();
	    result->setSegmentation(timeLocal.getTimeSec());


		timeLocal.reset(); timeLocal.start();
		//-----------------DETECTION OF REPRESENTATIVE FRAMES--------------------//
		result->setDescriptor(argv[2]);
		FeaturesGPU featGPU(argv[2], frameDir, featuresDir, segGPU.shots, segGPU.histograms);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setFeatDetecDesc(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//------------------------BAG OF VISUAL WORDS----------------------------//
		ClusteringGPU lbg(featuresDir);
		lbg.callLBG(150);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setLbgClustering(timeLocal.getTimeSec());

		//cout << "end" << endl;



		timeLocal.reset(); timeLocal.start();
		//-------------------HISTOGRAM OF VISUAL WORDS---------------------------//
		lbg.codeWordsHistogram(150,featuresDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setHistogramWords(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//-------------------VISUAL WORD VECTORS CLUSTERING----------------------//
		vector<int> indexCenters = lbg.clusterCodeWordsVector(segGPU.nscenes);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setWordsClutering(timeLocal.getTimeSec());




		//---------------------GENERATE SUMMARY----------------------------------//
		vector<int> vetorRepresFrames;
		for(int ki=0; ki<(int)featGPU.representativeFrames.size(); ki++)
		{
			for(int j=0; j<(int)featGPU.representativeFrames[ki].size(); j++)
			{
				vetorRepresFrames.push_back(featGPU.representativeFrames[ki][j]);
			}
		}

		vector<string> frames = FileOperations::listFiles(frameDir, ".jpg");
		sort(frames.begin(), frames.end());

		for(int ki=0; ki<(int)indexCenters.size(); ki++)
		{
			if(indexCenters[ki] >= vetorRepresFrames.size() || vetorRepresFrames[indexCenters[ki]] >= frames.size())
				continue;
			FileOperations::copyFile(frames[vetorRepresFrames[indexCenters[ki]]], summaryDir);
		}
		//-----------------------------------------------------------------------//


		timeLocal.reset(); timeLocal.start();
		//------------------------------FILTERING--------------------------------//
		FeaturesGPU::filterFrames(summaryDir);
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

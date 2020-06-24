#include <iostream>
#include <vector>
#include <iomanip>

#include "Defines.h"
#include "FileOperations.h"
#include "Results.h"

#include "DecoderMULT.h"
#include "SegmentMULT.h"
#include "FeaturesMULT.h"
#include "ClusteringMULT.h"

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

	omp_set_num_threads(omp_get_max_threads());

	for(int i=0; i<(int)videoNames.size(); i++)
	{
		cout << "Processing video " << videoNames[i] << "..." << endl;

		vector<string> splitStr0 = FileOperations::split(FileOperations::getSimpleName(videoNames[i]), '_');
		if(atoi(splitStr0[1].c_str()) > 5)
			continue;

		//-----------------------CREATE DIRECTORIES------------------------------//
		string frameDir = "frames/";
		string featuresDir = "features/";

		string dname(argv[2]);
		string summaryDir = "summaryOMP-"+dname+"-"+FileOperations::getSimpleName(videoNames[i])+"/";

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
		result->setArch("mult-omp");
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
		DecoderMULT::saveFramesOMP(videoNames[i], frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//------------------------VIDEO SEGMENTATION-----------------------------//
		SegmentMULT segMULT(frameDir, "omp");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setSegmentation(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//-----------------DETECTION OF REPRESENTATIVE FRAMES--------------------//
		result->setDescriptor(argv[2]);
		FeaturesMULT featMULT(argv[2], frameDir, featuresDir, segMULT.shots, segMULT.histograms, "omp");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setFeatDetecDesc(timeLocal.getTimeSec());



		timeLocal.reset(); timeLocal.start();
		//------------------------BAG OF VISUAL WORDS----------------------------//
		ClusteringMULT lbg(featuresDir);
		lbg.callLBG(150, "omp");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setLbgClustering(timeLocal.getTimeSec());


		timeLocal.reset(); timeLocal.start();
		//-------------------HISTOGRAM OF VISUAL WORDS---------------------------//
		lbg.codeWordsHistogram(150,featuresDir,"omp");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setHistogramWords(timeLocal.getTimeSec());



		timeLocal.reset(); timeLocal.start();
		//-------------------VISUAL WORD VECTORS CLUSTERING----------------------//
		vector<int> indexCenters = lbg.clusterCodeWordsVector(segMULT.nscenes);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setWordsClutering(timeLocal.getTimeSec());



		//---------------------GENERATE SUMMARY----------------------------------//
		vector<int> vetorRepresFrames;
		for(int ki=0; ki<(int)featMULT.representativeFrames.size(); ki++)
		{
			for(int j=0; j<(int)featMULT.representativeFrames[ki].size(); j++)
			{
				cout << featMULT.representativeFrames[ki][j] << endl;
				vetorRepresFrames.push_back(featMULT.representativeFrames[ki][j]);
			}
		}

		cout << "copying" << endl;
		vector<string> frames = FileOperations::listFiles(frameDir, ".jpg");
		for(int ki=0; ki<(int)indexCenters.size(); ki++)
		{
			if(indexCenters[ki] >= (int)vetorRepresFrames.size() || vetorRepresFrames[indexCenters[ki]] >= (int)frames.size())
				continue;
			FileOperations::copyFile(frames[vetorRepresFrames[indexCenters[ki]]], summaryDir);
		}
		//-----------------------------------------------------------------------//

		cout << "filter" << endl;

		timeLocal.reset(); timeLocal.start();
		//------------------------------FILTERING--------------------------------//
		FeaturesMULT::filterFrames(summaryDir);
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



		///////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////



		cout << "Processing video " << videoNames[i] << "..." << endl;
		//-----------------------CREATE DIRECTORIES------------------------------//
		frameDir = "frames/";
		featuresDir = "features/";

		//string dname(argv[2]);
		summaryDir = "summaryTHR-"+dname+"-"+FileOperations::getSimpleName(videoNames[i])+"/";

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
		result->setArch("mult-thr");
		result->setVideoName(FileOperations::getSimpleName(videoNames[i]));

		splitStr = FileOperations::split(FileOperations::getSimpleName(videoNames[i]), '_');

		result->setLength(atoi(splitStr[1].c_str()));
		result->setResolution(splitStr[3]);
		//-----------------------------------------------------------------------//




		timeGlobal.reset(); timeGlobal.start();
		timeLocal.reset(); timeLocal.start();
		//------------------------VIDEO DECODING---------------------------------//
		DecoderMULT::saveFrames(videoNames[i], frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//------------------------VIDEO SEGMENTATION-----------------------------//
		SegmentMULT segMULT1(frameDir, "thr");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setSegmentation(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//-----------------DETECTION OF REPRESENTATIVE FRAMES--------------------//
		result->setDescriptor(argv[2]);
		FeaturesMULT featMULT1(argv[2], frameDir, featuresDir, segMULT1.shots, segMULT1.histograms, "thr");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setFeatDetecDesc(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//------------------------BAG OF VISUAL WORDS----------------------------//
		ClusteringMULT lbg1(featuresDir);
		lbg1.callLBG(150, "thr");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setLbgClustering(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//-------------------HISTOGRAM OF VISUAL WORDS---------------------------//
		lbg1.codeWordsHistogram(150,featuresDir, "thr");
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setHistogramWords(timeLocal.getTimeSec());




		timeLocal.reset(); timeLocal.start();
		//-------------------VISUAL WORD VECTORS CLUSTERING----------------------//
		indexCenters = lbg1.clusterCodeWordsVector(segMULT1.nscenes);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		timeGlobal.stop();
		result->setWordsClutering(timeLocal.getTimeSec());




		//---------------------GENERATE SUMMARY----------------------------------//
		vetorRepresFrames.clear();
		for(int i=0; i<(int)featMULT1.representativeFrames.size(); i++)
		{
			for(int j=0; j<(int)featMULT1.representativeFrames[i].size(); j++)
			{
				vetorRepresFrames.push_back(featMULT1.representativeFrames[i][j]);
			}
		}

		frames = FileOperations::listFiles(frameDir, ".jpg");
		for(int i=0; i<(int)indexCenters.size(); i++)
		{
			if(indexCenters[i] >= vetorRepresFrames.size() || vetorRepresFrames[indexCenters[i]] >= frames.size())
							continue;
			FileOperations::copyFile(frames[vetorRepresFrames[indexCenters[i]]], summaryDir);
		}
		//-----------------------------------------------------------------------//


		timeLocal.reset(); timeLocal.start();
		//------------------------------FILTERING--------------------------------//
		FeaturesMULT::filterFrames(summaryDir);
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

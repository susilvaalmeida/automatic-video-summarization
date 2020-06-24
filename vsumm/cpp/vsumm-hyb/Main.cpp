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
#include "DecoderMULT.h"
#include "HistogramGPU.cuh"
#include "ClusteringGPU.cuh"
#include "Results.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <thread>
#include <string>
#include <omp.h>

using namespace std;



int main(int argc, char **argv) {

	omp_set_num_threads(omp_get_max_threads());

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
		string frameDir = "frames/";
		string summaryDir = "summaryHYB-"+FileOperations::getSimpleName(videoNames[i])+"/";

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
		//------------------------VIDEO SEGMENTATION-----------------------------//
		DecoderMULT::saveFramesOMP(videoNames[i], frameDir);
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());



		vector<string> frameNames = FileOperations::listFiles(frameDir, ".jpg");
		sort(frameNames.begin(), frameNames.end());


		//------------------------FEATURE EXTRACTION-----------------------------//
		vector<HistogramGPU> frameHistograms;

		cv::Mat img = cv::imread(frameNames[0]);
		int rows = img.rows;
		int cols = img.cols;

		size_t totalSize = 0;// rows*cols*3*frameNames.size()*sizeof(unsigned char);



		size_t freeMem = 0;
		size_t totalMem = 0;
		cudaMemGetInfo(&freeMem, &totalMem);

		cout << "Memoria CUDA disponivel: " << freeMem/1048576 << "MB" << endl;

		timeLocal.reset(); timeLocal.start();
		
		cv::Mat imgref = cv::imread(frameNames[0]);
		int cont = 0;
		int frameInicial = 0;
		while(cont < frameNames.size())
		{
			//cudaDeviceReset();
			//cudaSetDevice(0);

			cv::Mat concat;
			int framesQnt = 0;
			while(totalSize < freeMem && cont < frameNames.size())
			{
				if(totalSize + (size_t)((imgref.total()*imgref.elemSize()) + (256*3*sizeof(int)) + (256*3*sizeof(float))) > freeMem)
					break;

				framesQnt++;
				cv::Mat img = cv::imread(frameNames[cont]);
				concat.push_back(img);

				totalSize += (size_t)((imgref.total()*imgref.elemSize()) + (256*3*sizeof(int)) + (256*3*sizeof(float))); //image, histogram, variances size
				cont++;
			}

			
			vector<HistogramGPU> hists_temp = FeaturesGPU::computeAllHist(concat.data, frameInicial, framesQnt, rows, cols );
			

			frameHistograms.insert(frameHistograms.end(), hists_temp.begin(), hists_temp.end());

			//cout << "all histograms size:" << frameHistograms.size() << endl;

			frameInicial = cont-1;
			totalSize = 0;

		}
		//-----------------------------------------------------------------------//

		timeLocal.stop(); timeT += timeLocal.getTimeSec();
		result->setFeatExtraction(timeT);
		result->setFeatExtractionParallelPart(timeT);


		FeaturesMULT feat(frameNames,"omp");
		vector<HistogramMULT> frameHistograms = feat.getAllHist();
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setFeatExtraction(timeLocal.getTimeSec());





		timeLocal.reset(); timeLocal.start();
		//------------------------FRAMES CLUSTERING------------------------------//
		ClusteringGPU clust(frameHistograms);
		clust.kmeansOMP();
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
		for(int i=0; i<(int)finalKeyframes.size(); i++)
			FileOperations::copyFile(frameNames[finalKeyframes[i].getIdFrame()], summaryDir);

		FileOperations::deleteDir(frameDir);
		//-----------------------------------------------------------------------//

		timeGlobal.stop();
		result->setTotal(timeGlobal.getTimeSec());



		cout << "Summary generated. Saving results..." << endl;
		cout << timeGlobal.getTimeSec() << endl;
		result->save();
		cout << endl;

	}
		
	return 0;
}

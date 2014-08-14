
#include <iostream>
#include <vector>
#include <iomanip>


#include "Defines.h"
#include "FileOperations.h"
#include "DecoderGPU.h"
#include "HistogramGPU.cuh"
#include "ClusteringGPU.cuh"
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

	for(int i=0; i<videoNames.size(); i++)
	{
		cout << "Processing video " << videoNames[i] << "..." << endl;

		string dectype = argv[2];

		//-----------------------CREATE DIRECTORIES------------------------------//
		string frameDir = "framesGPU-dec"+dectype+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";
		string summaryDir = "summaryGPU-dec"+dectype+"-"+FileOperations::getSimpleName(videoNames[i])+"\\";

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
		result->setArch("gpu-"+dectype);
		result->setVideoName(FileOperations::getSimpleName(videoNames[i]));

		vector<string> splitStr = FileOperations::split(FileOperations::getSimpleName(videoNames[i]), '_');

		result->setLength(atoi(splitStr[1].c_str()));
		if(atoi(splitStr[1].c_str()) == 30)
			result->setResolution(splitStr[2]);
		else
			result->setResolution(splitStr[3]);
		//-----------------------------------------------------------------------//




		vector<HistogramGPU> frameHistograms;

		timeGlobal.reset(); timeGlobal.start();
		timeLocal.reset(); timeLocal.start();
		//------------------------VIDEO SEGMENTATION-----------------------------//


		if(dectype == "1")
			DecoderGPU::saveFrames(videoNames[i], frameDir);
		else if(dectype == "2")
			DecoderGPU::saveFramesGPU(videoNames[i], frameDir);
		else if(dectype == "3")
		{
			frameHistograms = DecoderGPU::saveFramesAndComputeHistGPU(videoNames[i], frameDir);
			timeLocal.stop();
			result->setFeatExtraction(timeLocal.getTimeSec());
		}
		//-----------------------------------------------------------------------//
		timeLocal.stop();
		result->setDecode(timeLocal.getTimeSec());


		vector<string> frameNames;
		frameNames = FileOperations::listFiles(frameDir, ".jpg");
		sort(frameNames.begin(), frameNames.end());


		cv::TickMeter time; double timeCut = 0.0; double timet = 0.0;


		if(dectype == "1" || dectype == "2")
		{

			time.reset(); time.start();

			//------------------------FEATURE EXTRACTION-----------------------------//
			cv::Mat img = cv::imread(frameNames[0]);
			int rows = img.rows;
			int cols = img.cols;

			size_t totalSize = 0;// rows*cols*3*frameNames.size()*sizeof(unsigned char);
			size_t freeMem = 838860800;
	

			cv::Mat imgref = cv::imread(frameNames[0]);
			int cont = 0;
			int frameInicial = 0;

			timeLocal.reset(); timeLocal.start();
			while(cont < frameNames.size())
			{
				
				cv::Mat concat;
				int framesQnt = 0;
				while(totalSize < freeMem && cont < frameNames.size())
				{
					if(totalSize + (size_t)((imgref.total()*imgref.elemSize()) + (256*sizeof(int)) + (256*sizeof(float))) > freeMem)
						break;

					framesQnt++;
					cv::Mat img = cv::imread(frameNames[cont]);
					concat.push_back(img);

					totalSize += (size_t)((imgref.total()*imgref.elemSize()) + (256*sizeof(int)) + (256*sizeof(float))); //image, histogram, variances size
					cont++;
				}
			

				
				vector<HistogramGPU> hists_temp = FeaturesGPU::computeAllHist(concat.data, frameInicial, framesQnt, rows, cols );
				
				
				frameHistograms.insert(frameHistograms.end(), hists_temp.begin(), hists_temp.end());


				//cout << "all histograms size:" << frameHistograms.size() << endl;

				frameInicial = cont-1;
				totalSize = 0;
				
			}
			//-----------------------------------------------------------------------//
			timeLocal.stop(); timet += timeLocal.getTimeSec();
			result->setFeatExtraction(timet);

		}


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


		result->print();
		result->save();
		cout << endl;

		//cudaDeviceReset();
		//exit(0);
	}

	return 0;
}

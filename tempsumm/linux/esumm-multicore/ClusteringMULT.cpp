#include <cmath>
#include "FileOperations.h"
#include "ClusteringMULT.h"
#include "MathOperationsMULT.h"
#include "Results.h"

#include <iostream>
#include <thread>
#include <omp.h>

ClusteringMULT::ClusteringMULT()
{
}

ClusteringMULT::ClusteringMULT(string featDir)
{
	string command = "cat " + featDir + "*.feat >> " + featDir + "allFeats.feat";
	system(command.c_str());

	dataLBG.clear();
	FileOperations::readMatFile(dataLBG, featDir + "allFeats.feat");
}

void ClusteringMULT::callXmeans(string ID, string parameterK, string maxLeafSize, string minBoxWidth,
		string cutFactor, string maxIterations, string numSplits, string maxCenters)
{
	changeEqualCols(dataXmeans);

	string content;
	for(int i=0; i<(int)dataXmeans[0].size(); i++)
	{
		std::ostringstream o;
		o << i;
		content += "x" + o.str() + " ";
	}
	content += "\n";

	for(int i=0; i<(int)dataXmeans.size(); i++)
	{
		for(int j=0; j<(int)dataXmeans[i].size(); j++)
		{
			std::ostringstream o;
			o.precision(6);
			o  << (float)dataXmeans[i][j];
			content +=  o.str() + " ";
		}
		content += "\n";
	}

	//cout << content << endl;

	FileOperations::createFile("matlabintro" + ID +".unit.ds", content);

	string comando;
	//system("pwd");
	comando = "./kmeans makeuni in matlabintro" + ID +".unit.ds";
	system(comando.c_str());

	comando = "./kmeans kmeans -k "+parameterK+" -method blacklist -max_leaf_size "+
			maxLeafSize+" -min_box_width "+minBoxWidth+" -cutoff_factor "+cutFactor+" -max_iter "+
			maxIterations+" -num_splits "+numSplits+" -max_ctrs "+maxCenters+
			" -in matlabintro" + ID +".unit.ds -save_ctrs ctrs" + ID+ ".out";
	system(comando.c_str());

	centersXmeans = FileOperations::readCtrsFile("ctrs" +ID+".out");

	FileOperations::deleteFile("matlabintro" + ID +".unit.ds.universe");
	FileOperations::deleteFile("matlabintro" + ID +".unit.ds");
	FileOperations::deleteFile("ctrs" + ID +".out");

}

void ClusteringMULT::changeEqualCols(vector<vector<int>>& features)
{
	vector<int> indicesIguais;
	for(int j=0; j<(int)features[0].size(); j++)
	{
		int first = features[0][j];

		int igual = 0;
		for(int k=1; k<(int)features.size(); k++)
		{
			if(first == features[k][j])
				igual++;
		}

		if(igual >= (int)features.size()-1)
			indicesIguais.push_back(j);
	}

	for(int j=0; j<(int)indicesIguais.size(); j++)
		features[0][j]++;
}

vector<int> ClusteringMULT::getClosestToCentroids()
{
	vector<int> closest;
	for(int i=0; i<(int)centersXmeans.size(); i++)
	{
		vector<float> distances0 = MathOperationsMULT::dist(dataXmeans, centersXmeans[i]);
		map<float,int> distances;
		for(int j=0; j<(int)distances0.size(); j++)
		{
			distances.insert(std::pair<float,int>(distances0[j],j));
		}
		closest.push_back(distances.begin()->second);
	}

	sort(closest.begin(), closest.end());
	return closest;
}

int ClusteringMULT::findNearestCluster(vector<float>& data, vector<vector<float>>& clusters, int clustersSize)
{
	int nearest = 0;
	float minDist = MathOperationsMULT::dist(data, clusters[0]);
	for(int i=1; i<clustersSize; i++)
	{
		float dist = MathOperationsMULT::dist(data, clusters[i]);
		if(dist < minDist)
		{
			minDist = dist;
			nearest = i;
		}
	}
	return nearest;
}

void ClusteringMULT::findClusterTH(int start, int end, vector<vector<float>>& clusters, int size, vector<int>& indexes)
{
	for(int i=start; i<end; i++)
	{
		int nearest = findNearestCluster(this->dataLBG[i], clusters, size);
		indexes[i] = nearest;
	}
}

void ClusteringMULT::findCodeWords(int start, int end, vector<vector<float>>& data, vector<vector<float>>& clusters, vector<int>& indexes)
{
	for(int i=start; i<end; i++)
	{
		int nearest = findNearestCluster(data[i], clusters, clusters.size());
		indexes[i] = nearest;
	}
}

void ClusteringMULT::callLBG(int qntCenters, string type)
{
	int max_num_threads = omp_get_max_threads();

	const float epsilon = 0.0510L;
	float distortionOld, distortionNew, diffDistortion;

	vector<float> distances;
	vector<float> distMin;

	int lenCenters = 1, numCentersGenerated = (int)pow(2, ceil(log10(qntCenters)/log10(2.0)));

	vector<vector<float>> centersTmp(numCentersGenerated);
	for(int i=0; i<numCentersGenerated; i++)
		centersTmp[i] = vector<float>(this->dataLBG[0].size(),0);

	MathOperationsMULT::meanCols(this->dataLBG, centersTmp[0]);

	distortionOld = distortionLBG(centersTmp[0]);
	diffDistortion = distortionOld;

	cv::TickMeter timeLocal; double time = 0.0;

	for(int countSplits = 0; countSplits < ceil(log10(qntCenters)/log10(2.0)); countSplits++)
	{
		//cout << "lgb interation " << countSplits << endl;
		for(int countC = 0; countC < lenCenters; countC++)
		{
			for(int i=0; i<(int)centersTmp[countC].size(); i++)
			{
				centersTmp[countC+lenCenters][i] = centersTmp[countC][i] * (1-epsilon);
				centersTmp[countC][i] = centersTmp[countC][i] * (1+epsilon);
			}
		}

		lenCenters *= 2;

		while(diffDistortion > epsilon)
		{
			timeLocal.reset(); timeLocal.start();
			vector<int> index(this->dataLBG.size());

			if(type == "omp")
			{
				int framesPerThread = this->dataLBG.size() / max_num_threads;

				omp_set_num_threads(max_num_threads);

				int id = 0;
#pragma omp parallel for
				for(int i=0; i<max_num_threads; i++)
				{
					findClusterTH(i*framesPerThread, (i*framesPerThread)+framesPerThread, centersTmp, lenCenters, index);
				}
			}
			else if(type == "thr")
			{
				int id = 0;
				int jump = this->dataLBG.size() / max_num_threads;

				vector<thread> threads;

				for(int i=0; i<max_num_threads; i++)
				{
					if(i==max_num_threads-1)
						threads.push_back(thread(&ClusteringMULT::findClusterTH, this, id, (int)this->dataLBG.size(), std::ref(centersTmp), lenCenters, std::ref(index)));
					else
						threads.push_back(thread(&ClusteringMULT::findClusterTH, this, id, id+jump, std::ref(centersTmp), lenCenters, std::ref(index)));
					id+=jump;
				}

				for(int i=0; i<max_num_threads; i++)
					threads[i].join();

			}


			meanInd(centersTmp, index, lenCenters);

			timeLocal.stop(); time += timeLocal.getTimeSec();
			distortionNew = distortionNewLBG(centersTmp, index);
			diffDistortion = (distortionOld - distortionNew) / distortionOld;
			distortionOld = distortionNew;
		}

		distortionOld = distortionNew;
		diffDistortion = distortionOld;
	}

	centersLBG.resize(qntCenters);
	for(int i=0; i<qntCenters; i++)
	{
		vector<float> aux(dataLBG[0].size());
		for(int j=0; j<(int)dataLBG[0].size(); j++)
			aux[j] = centersTmp[i][j];
		centersLBG[i] = aux;
	}

	Results *result;
	result = Results::getInstance();
	result->setLbgClusteringParallelPart(time);
	//cout << "centers: " << centersLBG.size() << endl;
}

float ClusteringMULT::distortionLBG(vector<float>& centeredData)
{
	float distortion = 0;
	for(int i=0; i<(int)dataLBG.size(); i++)
	{
		for(int j=0; j<(int)dataLBG[0].size(); j++)
			distortion += pow(dataLBG[i][j] - centeredData[j], 2);
	}
	return distortion;
}

float ClusteringMULT::distortionNewLBG(vector<vector<float>>& centeredData, vector<int>& index)
{
	float distortion = 0;
	for(int i=0; i<(int)dataLBG.size(); i++)
	{
		for(int j=0; j<(int)dataLBG[0].size(); j++)
			distortion += pow(dataLBG[i][j] - centeredData[index[i]][j], 2);
	}
	return distortion;
}

void ClusteringMULT::meanInd(vector<vector<float>>& meanData, vector<int>& index, int lenCenters)
{
	vector<int> tmpCont(lenCenters,0);

	for(int i=0; i<(int)this->dataLBG.size(); i++)
	{
		for (int j=0; j<(int)this->dataLBG[0].size(); j++)
			meanData[index[i]][j] += this->dataLBG[i][j];
		tmpCont[index[i]]++;
	}

	for(int i=0; i<lenCenters; i++)
	{
		if (tmpCont[i] != 0)
			for (int j=0; j<(int)this->dataLBG[0].size(); j++)
				meanData[i][j] /= tmpCont[i];
	}
}

void ClusteringMULT::codeWordsHistogram(int bins, string featDir, string type)
{
	int max_num_threads = omp_get_max_threads();

	if(dataLBG.size() == 0 || centersLBG.size() == 0)
		return;

	cv::TickMeter timeLocal; double time = 0.0;

	vector<string> feats = FileOperations::listFiles(featDir, ".feat");
	//sort(feats.begin(), feats.end());

	histogramVisualWords.resize(feats.size());

	for(int i=0; i<(int)feats.size(); i++)
	{
		vector<vector<float>> features;
		FileOperations::readMatFile(features, feats[i]);

		timeLocal.reset(); timeLocal.start();
		vector<int> index(features.size());

		if(type == "omp")
		{
			int framesPerThread = features.size() / max_num_threads;

			omp_set_num_threads(max_num_threads);

			int id = 0;
#pragma omp parallel for
			for(int i=0; i<max_num_threads; i++)
			{
				findCodeWords(i*framesPerThread, (i*framesPerThread)+framesPerThread, features, centersLBG, index);
			}
		}
		else if(type == "thr")
		{
			int id = 0;
			int jump = features.size() / max_num_threads;

			vector<thread> threads;

			for(int i=0; i<max_num_threads; i++)
			{
				if(i==max_num_threads-1)
					threads.push_back(thread(&ClusteringMULT::findCodeWords, this, id, (int)features.size(), std::ref(features), std::ref(centersLBG), std::ref(index)));
				else
					threads.push_back(thread(&ClusteringMULT::findCodeWords, this, id, id+jump, std::ref(features), std::ref(centersLBG), std::ref(index)));
				id+=jump;
			}

			for(int i=0; i<max_num_threads; i++)
				threads[i].join();

		}

		vector<int> histCodeWord(bins,0);
		for(int j=0; j<(int)index.size(); j++)
			histCodeWord[index[j]++]++;

		histogramVisualWords[i].resize(bins);
		histogramVisualWords[i] = histCodeWord;

		timeLocal.stop(); time += timeLocal.getTimeSec();

	}

	Results *result;
	result = Results::getInstance();
	result->setHistogramWordsParallelPart(time);
}

vector<int> ClusteringMULT::clusterCodeWordsVector(int nscenes)
{
	string nscenesSTR, maxnscenes;
	std::ostringstream o;
	o << nscenes;
	nscenesSTR = o.str();

	std::ostringstream o2;
	o2 << (nscenes*4);
	maxnscenes = o2.str();

	this->dataXmeans = this->histogramVisualWords;
	this->callXmeans("0",nscenesSTR,"1", "0.1","0.5","1000","4",maxnscenes);

	vector<int> ind = getClosestToCentroids();
	return ind;
}

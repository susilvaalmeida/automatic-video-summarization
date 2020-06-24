#include <cmath>
#include "FileOperations.h"
#include "ClusteringCPU.h"
#include "MathOperationsCPU.h"
#include "Results.h"

#include <iostream>


ClusteringCPU::ClusteringCPU()
{
}

ClusteringCPU::ClusteringCPU(string featDir)
{
	string command = "cat " + featDir + "*.feat >> " + featDir + "allFeats.feat";
	system(command.c_str());

	dataLBG.clear();
	FileOperations::readMatFile(dataLBG, featDir + "allFeats.feat");
}

void ClusteringCPU::callXmeans(string parameterK, string maxLeafSize, string minBoxWidth, 
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
			o  << dataXmeans[i][j];
			content +=  o.str() + " ";
		}
		content += "\n";
	}

	//cout << content << endl;

	FileOperations::createFile("matlabintro.unit.ds", content);

	string comando;
	//system("pwd");
	comando = "./kmeans makeuni in matlabintro.unit.ds";
	system(comando.c_str());

	comando = "sed -i 's/-nan -nan/2.000000 8.000000/g' matlabintro.unit.ds.universe";
	system(comando.c_str());

	comando = "./kmeans kmeans -k "+parameterK+" -method blacklist -max_leaf_size "+
			maxLeafSize+" -min_box_width "+minBoxWidth+" -cutoff_factor "+cutFactor+" -max_iter "+
			maxIterations+" -num_splits "+numSplits+" -max_ctrs "+maxCenters+
			" -in matlabintro.unit.ds -save_ctrs ctrs.out";
	system(comando.c_str());

	centersXmeans = FileOperations::readCtrsFile("ctrs.out");

	FileOperations::deleteFile("matlabintro.unit.ds.universe");
	FileOperations::deleteFile("matlabintro.unit.ds");
	FileOperations::deleteFile("ctrs.out");

}

void ClusteringCPU::changeEqualCols(vector<vector<int>>& features)
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

vector<int> ClusteringCPU::getClosestToCentroids()
{
	vector<int> closest;
	for(int i=0; i<(int)centersXmeans.size(); i++)
	{
		vector<float> distances0 = MathOperationsCPU::dist(dataXmeans, centersXmeans[i]);
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

int ClusteringCPU::findNearestCluster(vector<float>& data, vector<vector<float>>& clusters, int clustersSize)
{
	int nearest = 0;
	float minDist = MathOperationsCPU::dist(data, clusters[0]);
	for(int i=1; i<clustersSize; i++)
	{
		float dist = MathOperationsCPU::dist(data, clusters[i]);
		if(dist < minDist)
		{
			minDist = dist;
			nearest = i;
		}
	}
	return nearest;
}

void ClusteringCPU::callLBG(int qntCenters)
{
	cv::TickMeter timeLocal; double time = 0.0;


	const float epsilon = 0.0510L;
	float distortionOld, distortionNew = 0.0, diffDistortion;

	vector<float> distances;
	vector<float> distMin;

	int lenCenters = 1, numCentersGenerated = (int)pow(2, ceil(log10(qntCenters)/log10(2.0)));

	vector<vector<float>> centersTmp(numCentersGenerated);
	for(int i=0; i<numCentersGenerated; i++)
		centersTmp[i] = vector<float>(this->dataLBG[0].size(),0);

	MathOperationsCPU::meanCols(this->dataLBG, centersTmp[0]);

	distortionOld = distortionLBG(centersTmp[0]);
	diffDistortion = distortionOld;

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

			vector<int> index(this->dataLBG.size(), 0);

			for(int i=0; i<(int)this->dataLBG.size(); i++)
			{
				int nearest = findNearestCluster(this->dataLBG[i], centersTmp, lenCenters);
				index[i] = nearest;
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

float ClusteringCPU::distortionLBG(vector<float>& centeredData)
{
	float distortion = 0;
	for(int i=0; i<(int)dataLBG.size(); i++)
	{
		for(int j=0; j<(int)dataLBG[0].size(); j++)
			distortion += pow(dataLBG[i][j] - centeredData[j], 2);
	}
	return distortion;
}

float ClusteringCPU::distortionNewLBG(vector<vector<float>>& centeredData, vector<int>& index)
{
	float distortion = 0;
	for(int i=0; i<(int)dataLBG.size(); i++)
	{
		for(int j=0; j<(int)dataLBG[0].size(); j++)
			distortion += pow(dataLBG[i][j] - centeredData[index[i]][j], 2);
	}
	return distortion;
}

void ClusteringCPU::meanInd(vector<vector<float>>& meanData, vector<int>& index, int lenCenters)
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


void ClusteringCPU::codeWordsHistogram(int bins, string featDir)
{
	if(dataLBG.size() == 0 || centersLBG.size() == 0)
		return;

	cv::TickMeter localTime; double time = 0.0;

	vector<string> feats = FileOperations::listFiles(featDir, ".feat");
	//sort(feats.begin(), feats.end());

	histogramVisualWords.resize(feats.size());

	for(int i=0; i<(int)feats.size(); i++)
	{
		vector<vector<float>> features;
		FileOperations::readMatFile(features, feats[i]);

		localTime.reset(); localTime.start();

		vector<int> index(features.size(),0);

		for(int i=0; i<(int)features.size(); i++)
		{
			int nearest = findNearestCluster(features[i], centersLBG, centersLBG.size());
			index[i] = nearest;
		}

		vector<int> histCodeWord(bins,0);
		for(int j=0; j<(int)index.size(); j++)
			histCodeWord[index[j]++]++;

		histogramVisualWords[i].resize(bins);
		histogramVisualWords[i] = histCodeWord;

		localTime.stop(); time += localTime.getTimeSec();
	}

	Results *result;
	result = Results::getInstance();
	result->setHistogramWordsParallelPart(time);
}

vector<int> ClusteringCPU::clusterCodeWordsVector(int nscenes)
{
	string nscenesSTR, maxnscenes;
	std::ostringstream o;
	o << nscenes;
	nscenesSTR = o.str();

	std::ostringstream o2;
	o2 << (nscenes*4);
	maxnscenes = o2.str();

	this->dataXmeans = this->histogramVisualWords;
	this->callXmeans(nscenesSTR,"1", "0.1","0.5","1000","4",maxnscenes);

	vector<int> ind = getClosestToCentroids();
	return ind;
}

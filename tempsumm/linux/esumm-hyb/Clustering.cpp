/*
 * Clustering.cpp
 *
 *  Created on: Aug 1, 2014
 *      Author: suellen
 */



#include "ClusteringGPU.h"
#include <iostream>
#include <stdlib.h>
#include "FileOperations.h"
#include "MathOperations.h"

extern "C" vector<vector<float> > callLBGGPU(vector<vector<float> >& dataLBG, int qntCenters);
extern "C" vector<vector<int> > codeWordsHistogramGPU(string featDir, vector<vector<float> >& centers, int bins);

void ClusteringGPU::callLBG(int qntCenters)
{
	//cout << "call lbg gpu" << endl;
	centersLBG = callLBGGPU(dataLBG, qntCenters);
}

ClusteringGPU::ClusteringGPU()
{
}

ClusteringGPU::ClusteringGPU(string featDir)
{
	string command = "cat " + featDir + "*.feat >> " + featDir + "allFeats.feat";
	system(command.c_str());

	dataLBG.clear();
	FileOperations::readMatFile(dataLBG, featDir + "allFeats.feat");
}

void ClusteringGPU::callXmeans(string parameterK, string maxLeafSize, string minBoxWidth,
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
			o  << dataXmeans[i][j];
			content +=  o.str() + " ";
		}
		content += "\n";
	}

	//cout << content << endl;

	FileOperations::createFile("data.unit.ds", content);

	string comando;
	//system("pwd");
	comando = "./kmeans makeuni in data.unit.ds";
	system(comando.c_str());

	comando = "./kmeans kmeans -k "+parameterK+" -method blacklist -max_leaf_size "+
			maxLeafSize+" -min_box_width "+minBoxWidth+" -cutoff_factor "+cutFactor+" -max_iter "+
			maxIterations+" -num_splits "+numSplits+" -max_ctrs "+maxCenters+
			" -in data.unit.ds -save_ctrs ctrs.out";
	system(comando.c_str());

	centersXmeans = FileOperations::readCtrsFile("ctrs.out");

	FileOperations::deleteFile("data.unit.ds.universe");
	FileOperations::deleteFile("data.unit.ds");
	FileOperations::deleteFile("ctrs.out");

}

void ClusteringGPU::changeEqualCols(vector<vector<int> >& features)
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

vector<int> ClusteringGPU::getClosestToCentroids()
{
	vector<int> closest;
	for(int i=0; i<(int)centersXmeans.size(); i++)
	{
		vector<float> distances0 = MathOperations::dist(dataXmeans, centersXmeans[i]);
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

int ClusteringGPU::findNearestCluster(vector<float>& data, vector<vector<float> >& clusters, int clustersSize)
{
	int nearest = 0;
	float minDist = MathOperations::dist(data, clusters[0]);
	for(int i=1; i<clustersSize; i++)
	{
		float dist = MathOperations::dist(data, clusters[i]);
		if(dist < minDist)
		{
			minDist = dist;
			nearest = i;
		}
	}
	return nearest;
}

void ClusteringGPU::codeWordsHistogram(int bins, string featDir)
{
	histogramVisualWords = codeWordsHistogramGPU(featDir, centersLBG, bins);
}

vector<int> ClusteringGPU::clusterCodeWordsVector(int nscenes)
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


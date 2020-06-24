#include "ClusteringCPU.h"
#include "Results.h"

#include <vector>
#include <set>
#include <iterator>
#include <map>
#include <iostream>

#define BINS 16
using namespace std;

float ClusteringCPU::euclidianDist(HistogramCPU h1, HistogramCPU h2)
{
	float result = 0.0;
	for(int i=0; i<(int)h1.getHistogram().size(); i++)
		result += pow(h1.getHistPos(i) - h2.getHistPos(i), 2);
	return sqrt(result);
}

float ClusteringCPU::euclidianDist(HistogramCPU h1, vector<float> cluster)
{
	float result = 0.0;
	for(int i=0; i<BINS; i++)
	{
		result += pow(h1.getHistPos(i) - cluster[i], 2);
	}
	return sqrt(result);
}

float ClusteringCPU::euclidianDist(vector<float> h1, vector<float> h2)
{
	float result = 0.0;
	for(int i=0; i<BINS; i++)
		result += pow(h1[i] - h2[i], 2);
	return sqrt(result);
}
void ClusteringCPU::estimateK()
{
	double limiar = 0.4;
	int k_aux = 0;
	while(k_aux == 0 && limiar > -1)
	{
		for(int i=0; i<(int)features.size()-1; i++)
		{
			if(euclidianDist(features[i+1].getHistogramNorm(), features[i].getHistogramNorm()) > limiar)
				k_aux++;
		}
		if(k_aux == 0)
			limiar--;
	}
	this->k = k_aux;
	cout << k_aux << endl;
}

int ClusteringCPU::findNearestCluster(HistogramCPU hist)
{
	int nearest = 0;
	float minDist = euclidianDist(hist, clusters[0]);

	for(int i=1; i<(int)clusters.size(); i++)
	{
		float dist = euclidianDist(hist, clusters[i]);
		if(dist < minDist)
		{
			minDist = dist;
			nearest = i;
		}
	}
	return nearest;
}

void ClusteringCPU::kmeans()
{
	int bins = BINS;
	int threshold = 0;
	float delta;

	//estima k inicial
	estimateK();

	//inicializa vetores
	framesClass.resize(features.size(),-1);
	clusters.resize(k);
	for(int i=0; i<k; i++)
		clusters[i].resize(bins);

	vector<vector<float> > newClusters(k);
	for(int i=0; i<k; i++)
		newClusters[i].resize(bins);

	vector<int> newClusterSize(k,0);

	//escolhe primeiras features como centroides iniciais
	for(int i=0; i<k; i++)
	{
		for(int j=0; j<bins; j++)
			clusters[i][j] = features[i].getHistPos(j);
	}
	int loop = 0;
	do
	{
		delta = 0.0;

		for(int i=0; i<(int)features.size(); i++) {

			//encontra cluster mais proximo
			int nearest = findNearestCluster(features[i]);			
			if(framesClass[i] != nearest)
				delta += 1.0;

			//a feature i passa a pertencer ao cluster mais proximo
			framesClass[i] = nearest;

			//atualiza o centroide do cluster
			newClusterSize[nearest]++;
			for(int j=0; j<bins; j++)
				newClusters[nearest][j] += features[i].getHistPos(j);
		}

		//calcula a media dos novos centroides e atualiza clusters
		for(int i=0; i<k; i++) {
			for(int j=0; j<bins; j++) {
				if (newClusterSize[i] > 0)
					clusters[i][j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0;
			}
			newClusterSize[i] = 0;
		}
		delta /= features.size();
	}while(delta > threshold && loop++ < 500);
}

void ClusteringCPU::findKeyframes()
{
	//Encontra os frames com menor distancia para os outros frames do mesmo grupo
	for(int i=0; i<k; i++)
	{
		multimap<double, HistogramCPU> minDist;
		for(int j=0; j<(int)features.size(); j++)
		{
			if(framesClass[j] == i)
			{
				minDist.insert(pair<double, HistogramCPU>(euclidianDist(features[j],clusters[i]), features[j]));
			}
		}

		if(minDist.size() == 0)
			continue;

		std::multimap<double,HistogramCPU>::iterator it = minDist.begin();
		if((*it).first >= 0)
			keyframes.push_back((*it).second);
	}
}

bool sortfunction(HistogramCPU h1, HistogramCPU h2)
{
	return h1.getIdFrame() < h2.getIdFrame();
}

void ClusteringCPU::removeSimilarKeyframes()
{
	sort(keyframes.begin(), keyframes.end(), sortfunction);

	for(int i=0; i<(int)keyframes.size()-1; i++)
	{
		int j=i+1;
		while(j<(int)keyframes.size())
		{
			if(euclidianDist(keyframes[i].getHistogramNorm(), keyframes[j].getHistogramNorm())<0.3)
			{
				vector<HistogramCPU>::iterator it;
				it = keyframes.begin();
				advance(it,j);
				keyframes.erase(it);
				j--;
			}
			else
				j++;
		}
	}
}


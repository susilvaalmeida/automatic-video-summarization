#include "ClusteringMULT.h"
#include "Defines.h"
#include "Results.h"

#include <vector>
#include <iterator>
#include <map>
#include <iostream>
#include <omp.h>

#include <thread>



using namespace std;

float ClusteringMULT::euclidianDist(HistogramMULT h1, HistogramMULT h2)
{
	float result = 0.0;
	for(int i=0; i<BINS; i++)
		result += sqrt(pow(h1.getHistPos(i) - h2.getHistPos(i), 2));
	return result;
}

float ClusteringMULT::euclidianDist(HistogramMULT h1, vector<float> cluster)
{
	float result = 0.0;
	for(int i=0; i<BINS; i++)
		result += sqrt(pow(h1.getHistPos(i) - cluster[i], 2));
	return result;
}

float ClusteringMULT::euclidianDist(vector<float> h1, vector<float> h2)
{
	float result = 0.0;
	for(int i=0; i<BINS; i++)
		result += sqrt(pow(h1[i] - h2[i], 2));
	return result;
}
void ClusteringMULT::estimateK()
{
	double limiar = 2;  //0.5 para histograma normalizado
	int k_aux = 0;
	while(k_aux == 0 && limiar > -1)
	{
		for(int i=0; i<(int)featuresM.size()-1; i++)
		{
			//cout << euclidianDist(featuresM[i+1].getHistogramNorm(), featuresM[i].getHistogramNorm()) << endl;
			if(euclidianDist(featuresM[i+1].getHistogramNorm(), featuresM[i].getHistogramNorm()) > limiar)
				k_aux++;
		}
		if(k_aux == 0)
			limiar--;
	}
	this->k = k_aux;
}

int ClusteringMULT::findNearestCluster(HistogramMULT hist)
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

int ClusteringMULT::findNearestCluster(HistogramMULT hist, vector<vector<float> > clusters)
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

void ClusteringMULT::findClusterTH(int start, int end, int &delta, vector<vector<float> >& newClusters, vector<int>& newClusterSize)
{
	for(int i=start; i<end; i++)
	{
		//encontra cluster mais proximo
		int nearest = findNearestCluster(featuresM[i]);

		if(framesClass[i] != nearest)
			delta += 1.0;

		//a feature i passa a pertencer ao cluster mais proximo
		framesClass[i] = nearest;

		//atualiza o centroide do cluster
		newClusterSize[nearest]++;
		for(int j=0; j<BINS; j++)
			newClusters[nearest][j] += featuresM[i].getHistPos(j);
	}
}

void ClusteringMULT::kmeans()
{
	int max_num_threads = omp_get_max_threads();

	int bins = BINS;
	int threshold = 0;
	int delta;

	//estima k inicial
	estimateK();

	//inicializa vetores
	framesClass.resize(featuresM.size(),-1);
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
			clusters[i][j] = featuresM[i].getHistPos(j);
	}

	int loop = 0;
	do
	{
		delta = 0.0;

		int id = 0;
		int jump = featuresM.size() / max_num_threads;

		vector<thread> threads;

		for(int i=0; i<max_num_threads; i++)
		{
			if(i==max_num_threads-1)
				threads.push_back(thread(&ClusteringMULT::findClusterTH, this, id, (int)featuresM.size(), std::ref(delta), std::ref(newClusters), std::ref(newClusterSize)));
			else
				threads.push_back(thread(&ClusteringMULT::findClusterTH, this, id, id+jump, std::ref(delta), std::ref(newClusters), std::ref(newClusterSize)));
			id+=jump;
		}

		for(int i=0; i<max_num_threads; i++)
			threads[i].join();

		//calcula a media dos novos centroides e atualiza clusters
		for(int i=0; i<k; i++) {
			for(int j=0; j<bins; j++) {
				if (newClusterSize[i] > 0)
					clusters[i][j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0;
			}
			newClusterSize[i] = 0;
		}

		delta /= featuresM.size();
	}while(delta > threshold && loop++ < 500);

}

void ClusteringMULT::kmeansOMP()
{
	int max_num_threads = omp_get_max_threads();

	int bins = BINS;
	int threshold = 0;
	int delta;

	vector<HistogramMULT> features = featuresM;

	//estima k inicial
	estimateK();

	//inicializa vetores
	vector<int> local_framesClass;
	vector<vector<float> > local_clusters;

	local_framesClass.resize(features.size(),-1);
	local_clusters.resize(k);
	for(int i=0; i<k; i++)
		local_clusters[i].resize(bins);


	vector<vector<float> > newClusters(k);
	for(int i=0; i<k; i++)
		newClusters[i].resize(bins);

	vector<vector<vector<float> > > local_newClusters(max_num_threads);
	for(int i=0; i<max_num_threads; i++)
	{
		local_newClusters[i].resize(k);
		for(int j=0; j<k; j++)
			local_newClusters[i][j].resize(bins);
	}

	vector<int> newClusterSize(k,0);

	vector<vector<int> > local_newClusterSize(max_num_threads);
	for(int i=0; i<max_num_threads; i++)
	{
		local_newClusterSize[i].resize(k);
		for(int j=0; j<k; j++)
			local_newClusterSize[i][j] = 0;
	}


	//escolhe primeiras features como centroides iniciais
	for(int i=0; i<k; i++)
	{
		for(int j=0; j<bins; j++)
			local_clusters[i][j] = features[i].getHistPos(j);
	}

	int i,j,nearest;
	int loop = 0;
	do
	{
		delta = 0.0;

#pragma omp parallel shared(features,local_clusters,local_framesClass,local_newClusters,local_newClusterSize)
		{
			int tid = omp_get_thread_num();

#pragma omp for private(i,j,nearest) schedule(static) reduction(+:delta)
			for(i=0; i<(int)features.size(); i++)
			{
				//encontra cluster mais proximo
				nearest = findNearestCluster(features[i],local_clusters);

				if(local_framesClass[i] != nearest)
					delta += 1.0;

				//a feature i passa a pertencer ao cluster mais proximo
				local_framesClass[i] = nearest;

				//atualiza o centroide do cluster
				local_newClusterSize[tid][nearest]++;
				for(j=0; j<bins; j++)
					local_newClusters[tid][nearest][j] += features[i].getHistPos(j);
			}
		}

		for(i=0; i<k;i++)
		{
			for(j=0; j<max_num_threads; j++)
			{
				newClusterSize[i] += local_newClusterSize[j][i];
				local_newClusterSize[j][i] = 0.0;
				for(int k=0; k<bins; k++)
				{
					newClusters[i][k] += local_newClusters[j][i][k];
					local_newClusters[j][i][k] = 0.0;
				}
			}
		}

		//calcula a media dos novos centroides e atualiza clusters
		for(i=0; i<k; i++) {
			for(j=0; j<bins; j++) {
				if (newClusterSize[i] > 1)
					local_clusters[i][j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0;
			}
			newClusterSize[i] = 0;
		}

		delta /= features.size();
	}while(delta > threshold && loop++ < 500);

	clusters = local_clusters;
	framesClass = local_framesClass;

}

void ClusteringMULT::findKeyframes()
{
	//Encontra os frames com menor distancia para os outros frames do mesmo grupo
	for(int i=0; i<k; i++)
	{
		multimap<double, HistogramMULT> minDist;
		for(int j=0; j<(int)featuresM.size(); j++)
		{
			if(framesClass[j] == i)
				minDist.insert(pair<double, HistogramMULT>(euclidianDist(featuresM[j],clusters[i]), featuresM[j]));
		}

		if(minDist.size() == 0)
			continue;

		std::multimap<double,HistogramMULT>::iterator it = minDist.begin();
		if((*it).first >= 0)
			keyframes.push_back((*it).second);
	}

	//cout << "keyframes: " << keyframes.size() << endl;
}

bool sortfunction(HistogramMULT h1, HistogramMULT h2)
{
	return h1.getIdFrame() < h2.getIdFrame();
}

void ClusteringMULT::removeSimilarKeyframes()
{
	sort(keyframes.begin(), keyframes.end(), sortfunction);

	for(int i=0; i<(int)keyframes.size()-1; i++)
	{
		for(int j=i+1; j<(int)keyframes.size(); j++)
		{
			if(euclidianDist(keyframes[i].getHistogramNorm(), keyframes[j].getHistogramNorm())<0.5)
			{
				vector<HistogramMULT>::iterator it;
				it = keyframes.begin();
				advance(it,j);
				keyframes.erase(it);
			}
		}
	}
}


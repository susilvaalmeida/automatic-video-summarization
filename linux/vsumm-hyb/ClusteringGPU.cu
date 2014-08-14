#include "ClusteringGPU.cuh"
#include "Defines.h"
#include "Results.h"

#include <vector>
#include <iterator>
#include <map>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
		name = (type **)malloc(xDim * sizeof(type *));          \
		assert(name != NULL);                                   \
		name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
		assert(name[0] != NULL);                                \
		for (size_t i = 1; i < xDim; i++)                       \
		name[i] = name[i-1] + yDim;                         \
} while (0)

using namespace std;

inline void checkCuda(cudaError_t e)
{
	if (e != cudaSuccess)
		err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
}

inline void checkLastCudaError()
{
	checkCuda(cudaGetLastError());
}

static inline int nextPowerOfTwo(int n)
{
	n--;

	n = n >>  1 | n;
	n = n >>  2 | n;
	n = n >>  4 | n;
	n = n >>  8 | n;
	n = n >> 16 | n;
	n = n >> 32 | n;    //  For 64-bit ints

	return ++n;
}

__host__ __device__ inline static
float euclidianDist(int numBins, int numFeats, int numClustersK, float* features, float* clusters, int featId, int clusterId)
{
	float ans=0.0;
	for(int i = 0; i < numBins; i++)
	{
		//if(featId == 1)
			//printf("(%f-%f)", features[numFeats * i + featId],clusters[numClustersK * i + clusterId]);

		ans += sqrt((features[numFeats * i + featId] - clusters[numClustersK * i + clusterId]) *
				(features[numFeats * i + featId] - clusters[numClustersK * i + clusterId]));
	}
	return(ans);
}


__global__ static
void findNearestCluster(int numBins, int numFeats, int numClustersK, float* features, float* deviceClusters,
		int* frameClass, int* intermediates)
{
	extern __shared__ char sharedMemory[];
	unsigned char *frameClassChanged = (unsigned char *)sharedMemory;

	float *clusters = deviceClusters;

	frameClassChanged[threadIdx.x] = 0;
	int featId = blockDim.x * blockIdx.x + threadIdx.x;

	if(featId < numFeats)
	{
		float dist, min_dist;

		int index = 0;
		min_dist = euclidianDist(numBins, numFeats, numClustersK, features, clusters, featId, 0);

		for(int i=1; i<numClustersK; i++)
		{
			dist = euclidianDist(numBins, numFeats, numClustersK, features, clusters, featId, i);
			if (dist < min_dist)
			{
				min_dist = dist;
				index = i;
			}
		}

		if (frameClass[featId] != index)
			frameClassChanged[threadIdx.x] = 1;

		frameClass[featId] = index;

		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
		{
			if (threadIdx.x < s)
				frameClassChanged[threadIdx.x] += frameClassChanged[threadIdx.x + s];
			__syncthreads();
		}

		if (threadIdx.x == 0)
			intermediates[blockIdx.x] = frameClassChanged[0];
	}
}

__global__ static
void compute_delta(int *deviceIntermediates, int numIntermediates, int numIntermediates2)
{
	extern __shared__ unsigned int intermediates[];
	intermediates[threadIdx.x] = (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

	__syncthreads();

	for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
			intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		deviceIntermediates[0] = intermediates[0];
}

float ClusteringGPU::euclidianDist(vector<float> h1, vector<float> h2)
{
	float result = 0.0;
	for(int i=0; i<BINS; i++)
		result += sqrt(pow(h1[i] - h2[i], 2));
	return result;
}

float ClusteringGPU::euclidianDist(HistogramGPU h1, vector<float> cluster)
{
	float result = 0.0;
	for(int i=0; i<BINS; i++)
		result += sqrt(pow(h1.getHistPos(i) - cluster[i], 2));
	return result;
}

void ClusteringGPU::estimateK()
{
	double limiar = 1;  //0.5 para histograma normalizado
	int k_aux = 0;
	while(k_aux == 0 && limiar > -1)
	{
		for(int i=0; i<(int)features.size()-1; i++)
		{
			//cout << euclidianDist(features[i+1].getHistogramNorm(), features[i].getHistogramNorm()) << endl;
			if(euclidianDist(features[i+1].getHistogramNorm(), features[i].getHistogramNorm()) > limiar)
				k_aux++;
		}
		if(k_aux == 0)
			limiar--;
	}
	this->k = k_aux;
}

void ClusteringGPU::kmeansGPU()
{
	double time = 0.0, time2 = 0.0;
	cv::TickMeter timeLocal, timeLocal2;



	int bins = BINS;
	int threshold = 0;
	int delta, index;

	//estima k inicial
	estimateK();

	//inicializa vetores
	float** featuresInv;
	float** clustersInv;
	int* frameClassP;
	int* newClusterSize;
	float** newClusters;

	float *deviceFeats;
	float *deviceClusters;
	int *deviceFrameClass;
	int *deviceIntermediates;


	malloc2D(featuresInv,bins,features.size(),float);
	for(int i=0; i<bins; i++)
	{
		for(int j=0; j<features.size(); j++)
			featuresInv[i][j] = features[j].getHistPos(i);
	}

	malloc2D(clustersInv,bins,k,float);
	for(int i=0; i<bins; i++)
	{
		for(int j=0; j<k; j++)
			clustersInv[i][j] = featuresInv[i][j];
	}

	frameClassP = (int*)malloc(features.size()*sizeof(int));
	for(int i=0; i<features.size(); i++)
		frameClassP[i] = -1;

	newClusterSize = (int*)malloc(k*sizeof(int));
	for(int i=0; i<k; i++)
		newClusterSize[i] = 0;

	malloc2D(newClusters,bins,k,float);
	memset(newClusters[0],0,bins*k*sizeof(float));

	timeLocal.reset(); timeLocal.start();
	timeLocal2.reset(); timeLocal2.start();

	const unsigned int numThreadsPerClusterBlock = 128;
	const unsigned int numClusterBlocks = (features.size() + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
	const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned char);
	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
	const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);

	checkCuda(cudaMalloc(&deviceFeats, features.size()*bins*sizeof(float)));
	checkCuda(cudaMalloc(&deviceClusters, k*bins*sizeof(float)));
	checkCuda(cudaMalloc(&deviceFrameClass, features.size()*sizeof(int)));
	checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

	checkCuda(cudaMemcpy(deviceFeats, featuresInv[0], features.size()*bins*sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(deviceFrameClass, frameClassP, features.size()*sizeof(int), cudaMemcpyHostToDevice));

	timeLocal.stop(); time += timeLocal.getTimeSec();
	timeLocal2.stop(); time2 += timeLocal2.getTimeSec();

	int loop = 0;
	do
	{
		timeLocal2.reset(); timeLocal2.start();

		checkCuda(cudaMemcpy(deviceClusters, clustersInv[0], k*bins*sizeof(float), cudaMemcpyHostToDevice));

		timeLocal.reset(); timeLocal.start();

		findNearestCluster<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
				(bins, features.size(), k, deviceFeats, deviceClusters, deviceFrameClass, deviceIntermediates);
		cudaDeviceSynchronize(); checkLastCudaError();

		compute_delta <<<1,numReductionThreads,reductionBlockSharedDataSize>>>(deviceIntermediates,numClusterBlocks,numReductionThreads);
		cudaDeviceSynchronize(); checkLastCudaError();

		timeLocal.stop(); time += timeLocal.getTimeSec();


		int d;
		checkCuda(cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost));
		delta = (float)d;

		checkCuda(cudaMemcpy(frameClassP, deviceFrameClass, features.size()*sizeof(int), cudaMemcpyDeviceToHost));

		timeLocal2.stop(); time2 += timeLocal2.getTimeSec();

		for(int i=0; i<features.size(); i++)
		{
			index = frameClassP[i];
			newClusterSize[index]++;
			for(int j=0; j<bins; j++)
				newClusters[j][index] += features[i].getHistPos(j);
		}

		//calcula a media dos novos centroides e atualiza clusters
		for(int i=0; i<k; i++) {
			for(int j=0; j<bins; j++) {
				if (newClusterSize[i] > 0)
				{
					//cout << "new cluster: " << newClusters[j][i] << "-" << newClusterSize[i] << endl;
					clustersInv[j][i] = newClusters[j][i] / newClusterSize[i];
				}
				newClusters[j][i] = 0.0;
			}
			newClusterSize[i] = 0;
		}

		delta /= features.size();



		//cout << "delta: " << delta << " loop: " << loop << endl;
	}while (delta > threshold && loop++ < 500);



	framesClass.resize(features.size());
	for(int i=0; i<features.size(); i++)
	{
		framesClass[i] = frameClassP[i];
		//cout << framesClass[i] <<  " ";
	}
	//cout << endl;

	clusters.resize(k);
	for(int i=0; i<k; i++)
		clusters[i].resize(bins);

	for(int i=0; i<k; i++)
	{
		for(int j=0; j<bins; j++)
			clusters[i][j] = clustersInv[j][i];
	}



	Results *result;
	result = Results::getInstance();
	result->setClusteringGpuCopy(time);
	result->setClusteringParallelPart(time2);

	checkCuda(cudaFree(deviceFeats));
	checkCuda(cudaFree(deviceClusters));
	checkCuda(cudaFree(deviceFrameClass));
	checkCuda(cudaFree(deviceIntermediates));

	free(featuresInv[0]);
	free(featuresInv);
	free(clustersInv[0]);
	free(clustersInv);
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
}

void ClusteringGPU::findKeyframes()
{
	//Encontra os frames com menor distancia para os outros frames do mesmo grupo
	for(int i=0; i<k; i++)
	{
		multimap<double, HistogramGPU> minDist;
		for(int j=0; j<(int)features.size(); j++)
		{
			if(framesClass[j] == i)
				minDist.insert(pair<double, HistogramGPU>(euclidianDist(features[j],clusters[i]), features[j]));
		}

		if(minDist.size() == 0)
			continue;

		std::multimap<double,HistogramGPU>::iterator it = minDist.begin();
		if((*it).first >= 0)
			keyframes.push_back((*it).second);
	}

	//cout << "keyframes: " << keyframes.size() << endl;
}

bool sortfunction(HistogramGPU h1, HistogramGPU h2)
{
	return h1.getIdFrame() < h2.getIdFrame();
}

void ClusteringGPU::removeSimilarKeyframes()
{
	sort(keyframes.begin(), keyframes.end(), sortfunction);

	for(int i=0; i<(int)keyframes.size()-1; i++)
	{
		for(int j=i+1; j<(int)keyframes.size(); j++)
		{
			//cout << keyframes[i].getIdFrame() << "," << keyframes[j].getIdFrame() << "= "<< euclidianDist(keyframes[i].getHistogramNorm(), keyframes[j].getHistogramNorm()) << endl;
			if(euclidianDist(keyframes[i].getHistogramNorm(), keyframes[j].getHistogramNorm())<0.5)
			{
				vector<HistogramGPU>::iterator it;
				it = keyframes.begin();
				advance(it,j);
				keyframes.erase(it);
			}
		}
	}

	//cout << "keyframes: " << keyframes.size() << endl;
}


#include <cmath>
#include "FileOperations.h"
#include "MathOperations.h"
#include "Results.h"
#include <iostream>

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

inline void checkCuda(cudaError_t e)
{
	if (e != cudaSuccess)
		err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
}

inline void checkLastCudaError()
{
	checkCuda(cudaGetLastError());
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
void findNearestClusterGPU(int numBins, int numFeats, int numClustersK, float* features, float* deviceClusters,int* frameClass)
{
	float *clusters = deviceClusters;

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

		frameClass[featId] = index;
	}
}


extern "C"
vector<vector<float> > callLBGGPU(vector<vector<float> >& dataLBG, int qntCenters)
{
	Results *result;
	result = Results::getInstance();

	cv::TickMeter timeLocal, timeLocal2;
	double timeD = 0.0, timeP = 0.0;


	cudaDeviceReset();
	cudaSetDevice(0);

	const float epsilon = 0.0510L;
	float distortionOld, distortionNew, diffDistortion;

	vector<float> distances;
	vector<float> distMin;

	int lenCenters = 1, numCentersGenerated = (int)pow(2, ceil(log10(qntCenters)/log10(2.0)));

	vector<vector<float> > centersTmp(numCentersGenerated);
	for(int i=0; i<numCentersGenerated; i++)
		centersTmp[i] = vector<float>(dataLBG[0].size(),0);

	MathOperations::meanCols(dataLBG, centersTmp[0]);

	float distortion = 0;
	for(int i=0; i<(int)dataLBG.size(); i++)
	{
		for(int j=0; j<(int)dataLBG[0].size(); j++)
			distortion += pow(dataLBG[i][j] - centersTmp[0][j], 2);
	}
	distortionOld = distortion;//LBG(centersTmp[0]);
	diffDistortion = distortionOld;


	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////
	timeLocal.reset(); timeLocal.start();
	timeLocal2.reset(); timeLocal2.start();

	float** featuresInv;
	float** clustersInv;
	int* frameClassP;
	int* newClusterSize;
	float** newClusters;

	float *deviceFeats;
	float *deviceClusters;
	int *deviceFrameClass;

	int bins = dataLBG[0].size();

	malloc2D(featuresInv,bins,dataLBG.size(),float);
	for(int i=0; i<bins; i++)
	{
		for(int j=0; j<dataLBG.size(); j++)
			featuresInv[i][j] = dataLBG[j][i];
	}

	malloc2D(clustersInv,bins,numCentersGenerated,float);
	for(int i=0; i<bins; i++)
	{
		for(int j=0; j<numCentersGenerated; j++)
			clustersInv[i][j] = centersTmp[j][i];
	}

	frameClassP = (int*)malloc(dataLBG.size()*sizeof(int));
	for(int i=0; i<dataLBG.size(); i++)
		frameClassP[i] = -1;

	newClusterSize = (int*)malloc(numCentersGenerated*sizeof(int));
	for(int i=0; i<numCentersGenerated; i++)
		newClusterSize[i] = 0;

	malloc2D(newClusters,bins,numCentersGenerated,float);
	memset(newClusters[0],0,bins*numCentersGenerated*sizeof(float));


	timeLocal.stop(); timeD += timeLocal.getTimeSec();


	const unsigned int numThreadsPerClusterBlock = 128;
	const unsigned int numClusterBlocks = (dataLBG.size() + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
	const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned char);

	checkCuda(cudaMalloc(&deviceFeats, dataLBG.size()*bins*sizeof(float)));
	checkCuda(cudaMalloc(&deviceClusters, numCentersGenerated*bins*sizeof(float)));
	checkCuda(cudaMalloc(&deviceFrameClass, dataLBG.size()*sizeof(int)));

	checkCuda(cudaMemcpy(deviceFeats, featuresInv[0], dataLBG.size()*bins*sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(deviceFrameClass, frameClassP, dataLBG.size()*sizeof(int), cudaMemcpyHostToDevice));

	timeLocal2.stop(); timeP += timeLocal2.getTimeSec();


	for(int countSplits = 0; countSplits < ceil(log10(qntCenters)/log10(2.0)); countSplits++)
	{

		//cout << "lgb interation " << countSplits << endl;
		for(int countC = 0; countC < lenCenters; countC++)
		{
			for(int i=0; i<(int)centersTmp[countC].size(); i++)
			{
				//centersTmp[countC+lenCenters][i] = centersTmp[countC][i] * (1-epsilon);
				//centersTmp[countC][i] = centersTmp[countC][i] * (1+epsilon);
				clustersInv[i][countC+lenCenters] = clustersInv[i][countC] * (1-epsilon);
				clustersInv[i][countC] = clustersInv[i][countC] * (1+epsilon);
			}
		}

		lenCenters *= 2;


		//cout << "centers: " << lenCenters << endl;

		while(diffDistortion > epsilon)
		{
			//cout << "diff distortion: " << diffDistortion <<  " epsilon: " << epsilon << endl;
			timeLocal2.reset(); timeLocal2.start();

			checkCuda(cudaMemcpy(deviceClusters, clustersInv[0], lenCenters*bins*sizeof(float), cudaMemcpyHostToDevice));

			timeLocal.reset(); timeLocal.start();

			findNearestClusterGPU<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
					(bins, dataLBG.size(), lenCenters, deviceFeats, deviceClusters, deviceFrameClass);
			cudaDeviceSynchronize(); checkLastCudaError();

			timeLocal.stop(); timeD += timeLocal.getTimeSec();

			checkCuda(cudaMemcpy(frameClassP, deviceFrameClass, dataLBG.size()*sizeof(int), cudaMemcpyDeviceToHost));

			timeLocal2.stop(); timeP += timeLocal2.getTimeSec();

			/*vector<int> index(dataLBG.size(), 0);

			for(int i=0; i<dataLBG.size(); i++)
			{
				int nearest = findNearestCluster(dataLBG[i], centersTmp, lenCenters);
				index[i] = nearest;
			}
			 */

			//meanInd(centersTmp, index, lenCenters);

			for(int i=0; i<dataLBG.size(); i++)
			{
				int index = frameClassP[i];
				newClusterSize[index]++;
				for(int j=0; j<bins; j++)
					newClusters[j][index] += dataLBG[i][j];
			}

			//calcula a media dos novos centroides e atualiza clusters
			for(int i=0; i<lenCenters; i++) {
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


			float distortion = 0;
			for(int i=0; i<dataLBG.size(); i++)
			{
				for(int j=0; j<bins; j++)
					distortion += pow(featuresInv[j][i] - clustersInv[j][frameClassP[i]], 2);
			}

			distortionNew = distortion; //NewLBG(centersTmp, index);

			diffDistortion = abs(distortionOld - distortionNew) / distortionOld;
			distortionOld = distortionNew;


		}

		distortionOld = distortionNew;
		diffDistortion = distortionOld;
	}
	//cout << "aqui1" << endl;


	vector< vector<float> > centersLBG;

	for(int i=0; i<qntCenters; i++)
	{
		vector<float> v;
		for(int j=0; j<bins; j++)
		{
			//			cout << i << "," << j << " " << clustersInv[j][i] << endl;

			v.push_back(clustersInv[j][i]);
		}
		centersLBG.push_back(v);
	}

	result->setLbgClusteringWithoutGpuCopy(timeD);
	result->setLbgClusteringParallelPart(timeP);

	//cout << "aqui" << endl;
	cudaFree(deviceClusters);
	cudaFree(deviceFeats);
	cudaFree(deviceFrameClass);

	free(featuresInv[0]);
	free(featuresInv);
	free(clustersInv[0]);
	free(clustersInv);
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
	free(frameClassP);

	return centersLBG;
	//cout << "centers: " << centersLBG.size() << endl;
}

__global__ static
void findNearestClusterAndHistGPU(int numBins, int numFeats, int numClustersK, float* features, float* deviceClusters,int* histogram)
{
	float *clusters = deviceClusters;

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

		//histogram[featId] = index;
		atomicAdd(&(histogram[(index+1)]),1);
	}
}

extern "C"
vector<vector<int> > codeWordsHistogramGPU(string featDir, vector<vector<float> >& centers, int bins)
{
	cudaDeviceReset();
	cudaSetDevice(0);

	Results *result;
	result = Results::getInstance();

	cv::TickMeter timeLocal, timeLocal2;
	double timeD = 0.0, timeP = 0.0;

	float** clustersInv;
	float *deviceClusters;
	malloc2D(clustersInv,centers.size(),centers[0].size(),float);
	for(int i=0; i<centers[0].size(); i++)
	{
		for(int j=0; j<centers.size(); j++)
			clustersInv[i][j] = centers[j][i];
	}

	checkCuda(cudaMalloc(&deviceClusters, centers.size()*centers[0].size()*sizeof(float)));
	checkCuda(cudaMemcpy(deviceClusters, clustersInv[0], centers.size()*centers[0].size()*sizeof(float), cudaMemcpyHostToDevice));


	vector<string> feats = FileOperations::listFiles(featDir, ".feat");
	vector< vector<int > > histograms(feats.size());


	for(int i=0; i<feats.size(); i++)
	{
		vector<vector<float> > features;
		FileOperations::readMatFile(features, feats[i]);

		timeLocal.reset(); timeLocal.start();
		timeLocal2.reset(); timeLocal2.start();

		float** featuresInv;
		int* hist;

		float *deviceFeats;
		int *deviceHist;

		malloc2D(featuresInv,features.size(),features[0].size(),float);
		for(int i=0; i<features[0].size(); i++)
		{
			for(int j=0; j<features.size(); j++)
				featuresInv[i][j] = features[j][i];
		}

		hist = (int*)malloc(bins*sizeof(int));
		for(int i=0; i<bins; i++)
			hist[i] = 0;

		timeLocal.stop(); timeD += timeLocal.getTimeSec();

		const unsigned int numThreadsPerClusterBlock = 128;
		const unsigned int numClusterBlocks = (features.size() + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
		const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned char);

		checkCuda(cudaMalloc(&deviceFeats, features.size()*features[0].size()*sizeof(float)));
		checkCuda(cudaMemcpy(deviceFeats, featuresInv[0], features.size()*features[0].size()*sizeof(float), cudaMemcpyHostToDevice));

		checkCuda(cudaMalloc(&deviceHist, bins*sizeof(int)));
		checkCuda(cudaMemcpy(deviceHist, hist, bins*sizeof(int), cudaMemcpyHostToDevice));

		timeLocal.reset(); timeLocal.start();

		findNearestClusterAndHistGPU<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
				(features[0].size(), features.size(), centers.size(), deviceFeats, deviceClusters, deviceHist);

		checkCuda(cudaDeviceSynchronize());
		checkCuda(cudaGetLastError());

		timeLocal.stop(); timeD += timeLocal.getTimeSec();

		checkCuda(cudaMemcpy(hist, deviceHist, bins*sizeof(int), cudaMemcpyDeviceToHost));

		cudaFree(deviceFeats);
		cudaFree(deviceHist);
		free(hist);
		free(featuresInv[0]);
		free(featuresInv);

		timeLocal2.stop(); timeP += timeLocal2.getTimeSec();

		vector<int> histFinal(bins,0);
		for(int i=0; i<bins; i++)
		{
			histFinal[i] = hist[i];
			//cout << hist[i] << " ";
		}
		//cout << endl;
		histograms[i] = histFinal;

	}
	result->setHistogramWordsWithoutGpuCopy(timeD);
	result->setHistogramWordsParallelPart(timeP);

	free(clustersInv[0]);
	free(clustersInv);

	cudaFree(deviceClusters);
	return histograms;
}

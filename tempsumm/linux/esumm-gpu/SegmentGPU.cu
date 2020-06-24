#include "SegmentGPU.h"
#include "MathOperations.h"
#include <algorithm>
#include <iostream>

#include "Results.h"



using namespace std;

__global__ void gray_kernel(unsigned char* d_in, unsigned char* d_out, int linhas, int colunas, int frames){

	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	int idWrite = (yIndex + xIndex * colunas) + (linhas * colunas * zIndex) ;
	int idRead	= idWrite*3;

	if(xIndex < linhas && yIndex < colunas && zIndex < frames)
	{
		double gray_val = 0.299*d_in[idRead] + 0.587*d_in[idRead+1] + 0.114*d_in[idRead+2];
		d_out[idWrite] = (unsigned char)gray_val;
	}
}

__global__ void histograma_basic_kernel(unsigned char *img, unsigned int *histo,int linhas, int colunas, int n)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	int id = (yIndex + xIndex * colunas) + (linhas * colunas * zIndex) ;
	id = id*3;
	if(xIndex < linhas && yIndex < colunas && zIndex < n) //&& id < linhas*colunas*n)
	{
		unsigned char r=img[id+2], g=img[id+1], b=img[id];
		atomicAdd(&(histo[r+(256*3*zIndex)]),1);
		atomicAdd(&(histo[g+(256*3*zIndex)+256]),1);
		atomicAdd(&(histo[b+(256*3*zIndex)+2*256]),1);
	}
}

__global__ void mean_kernel(unsigned char*img, float*mean, int imgSize, int nImgs)
{
	long id = (blockDim.x * blockIdx.x + threadIdx.x) * imgSize;
	long idWrite = blockDim.x * blockIdx.x + threadIdx.x;

	mean[idWrite] = 0;
	if(id < imgSize*nImgs)
	{
		long count = id;
		float m = 0;
		for(int i=0; i<imgSize; i++)
			m = m + img[count++];

		//float media = m / imgSize;
		float media = __fdividef(m,imgSize);
		mean[idWrite] = media;
	}
}

__global__ void variance_kernel(unsigned char* img, float* var, int imgSize, int nImgs, float* mean)
{
	long id = (blockDim.x * blockIdx.x + threadIdx.x) * imgSize;
	long idWrite = blockDim.x * blockIdx.x + threadIdx.x;

	var[idWrite] = 0;
	if(id < imgSize*nImgs)
	{
		long count = id;
		float value = 0;
		for(int i=0; i<imgSize; i++)
		{
			value = value +  __powf(abs(img[count] - mean[idWrite]), 2);
			count++;
		}
		//float variancia = value / (imgSize-1);
		float variancia = __fdividef(value, imgSize-1);
		var[idWrite] = variancia;
	}
}

__global__ void mean_variance_kernel(unsigned char*img, float*variancias, int imgSize, int nImgs)
{
	long id = (blockDim.x * blockIdx.x + threadIdx.x) * imgSize;
	long idWrite = blockDim.x * blockIdx.x + threadIdx.x;

	if(id < imgSize*nImgs)
	{
		long count = id;
		float m = 0;
		for(int i=0; i<imgSize; i++)
			m = m + img[count++];

		//float media = m / imgSize;
		float media = __fdividef(m,imgSize);

		count = id;
		float value = 0;
		for(int i=0; i<imgSize; i++)
		{
			value = value +  __powf(abs(img[count] - media), 2);
			count++;
		}

		float variancia = __fdividef(value, imgSize-1);
		variancias[idWrite] = variancia;
	}
}

__global__ void cosineDissimilarity_kernel(unsigned int* histograms, float* results, int histSize, int nHists)
{
	long id = (blockDim.x * blockIdx.x + threadIdx.x) * histSize;
	long idWrite = blockDim.x * blockIdx.x + threadIdx.x;

	if(id < histSize*(nHists-1) && idWrite < nHists)
	{
		results[idWrite] = 0;

		long count = id;
		float dotProduct = 0, magnitudeH1 = 0, magnitudeH2 = 0;
		for(int i=0; i<histSize; i++)
		{
			dotProduct = dotProduct + (float) histograms[count] * (float) histograms[count+histSize];
			magnitudeH1 = magnitudeH1 + __powf((float)histograms[count],2);// * (float) histograms[count];
			magnitudeH2 = magnitudeH2 + __powf((float)histograms[count+histSize],2);// * (float) histograms[count+histSize];
			count++;
		}

		//float similarity = dotProduct / (sqrt(magnitudeH1) * sqrt(magnitudeH2));
		float similarity = __fdividef(dotProduct, __fsqrt_rn(magnitudeH1) * __fsqrt_rn(magnitudeH2));
		results[idWrite] = 1-similarity;
		//printf("write: %d\n", idWrite);
	}
}

//extern "C"
void computeHistVarDiss(unsigned char* images, int frameInicial, int qntFrames, int rows, int cols, vector<double>& variances, vector<double>& dissimilarity, vector<vector<int> >& histograms)
{
	//cout << "compute hist var " << frameInicial << endl;
	cudaDeviceReset();
	cudaSetDevice(0);

	cv::TickMeter time; double timeD = 0.0;

	time.reset(); time.start();
	cudaError_t cudaStatus;

	//cudaStream_t stream1, stream2;
	//cudaStreamCreate(&stream1);
	//cudaStreamCreate(&stream2);

	size_t totalSize = rows*cols*qntFrames*3+1;

	unsigned char *gpu_in, *gpu_out;
	cudaStatus = cudaMalloc((void**)&gpu_in, totalSize);
	if(cudaStatus != cudaSuccess) cout << "erro: malloc gpu_in " <<  cudaGetErrorString(cudaStatus) << endl;

	cudaStatus = cudaMalloc((void**)&gpu_out, totalSize/3);
	if(cudaStatus != cudaSuccess) cout << "erro: malloc gpu_out " <<  cudaGetErrorString(cudaStatus) << endl;

	time.stop(); timeD += time.getTimeSec();

	cudaStatus = cudaMemcpy(gpu_in, images, totalSize, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) cout << "erro: memcpy gpu_in " <<  cudaGetErrorString(cudaStatus) << endl;

	time.reset(); time.start();

	dim3 dimBlock(16,16,4);
	int blocksInX = (rows+16-1)/16;
	int blocksInY = (cols+16-1)/16;
	int blocksInZ = (qntFrames+4-1)/4;
	dim3 dimGrid(blocksInX, blocksInY, blocksInZ);

	//transform to gray to compute variance
	gray_kernel<<<dimGrid,dimBlock>>>(gpu_in, gpu_out, rows, cols, qntFrames);

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) cout << "erro: gray_kernel " <<  cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) cout << "erro: gray_kernel " <<  cudaGetErrorString(cudaStatus) << endl;

	float *var_gpu_out;
	cudaStatus = cudaMalloc((void**)&var_gpu_out, qntFrames*sizeof(float));
	if(cudaStatus != cudaSuccess) cout << "erro: malloc var_gpu_out " <<  cudaGetErrorString(cudaStatus) << endl;

	int tt = (qntFrames+32-1)/32;
	//compute mean for the variance calculation
	mean_variance_kernel<<<tt,32>>>(gpu_out, var_gpu_out, rows*cols, qntFrames);

	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) cout << "erro: kernel variance " <<  cudaGetErrorString(cudaStatus) << endl;

	//cout << "free gpu_out " <<endl;
	cudaFree(gpu_out);

	////////////////////////////////////////////////////////////////////////////

	size_t histogramsSize = qntFrames * 256 * 3;

	unsigned int *histGPU;
	cudaStatus = cudaMalloc((void**)&histGPU, histogramsSize*sizeof(unsigned int));
	if(cudaStatus != cudaSuccess) cout << "erro: malloc histGPU " <<  cudaGetErrorString(cudaStatus) << endl;

	time.stop(); timeD += time.getTimeSec();


	cudaStatus = cudaMemset(histGPU, 0, histogramsSize*sizeof(unsigned int));
	if(cudaStatus != cudaSuccess) cout << "erro: memset histGPU " <<  cudaGetErrorString(cudaStatus) << endl;

	time.reset(); time.start();

	//compute histograms
	histograma_basic_kernel<<<dimGrid,dimBlock>>>(gpu_in, histGPU, rows, cols, qntFrames);

	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) cout << "erro: histogram kernel " <<  cudaGetErrorString(cudaStatus) << endl;

	//cout << "free gpu_in " << endl;
	cudaFree(gpu_in);

	float *dissGPU;
	cudaStatus = cudaMalloc((void**)&dissGPU, qntFrames*sizeof(float));
	if(cudaStatus != cudaSuccess) cout << "erro: malloc dissGPU " <<  cudaGetErrorString(cudaStatus) << endl;

	time.stop(); timeD += time.getTimeSec();

	cudaStatus = cudaMemset(dissGPU, 0.0, qntFrames);
	if(cudaStatus != cudaSuccess) cout << "erro: memset dissGPU " <<  cudaGetErrorString(cudaStatus) << endl;

	time.reset(); time.start();

	//compute cosine dissimilarity
	cosineDissimilarity_kernel<<<tt,32>>>(histGPU, dissGPU, 256*3, qntFrames);

	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess)
		cout << "erro: kernel dissimilarity" << endl;

	unsigned int *hists_out = new unsigned int[histogramsSize];
	float *diss_out = new float[qntFrames];

	time.stop(); timeD += time.getTimeSec();



	//cudaStatus = cudaStreamSynchronize(stream1);
	//	if(cudaStatus != cudaSuccess) cout << "erro: stream2 " <<  cudaGetErrorString(cudaStatus) << endl;

	cudaStatus = cudaMemcpy(hists_out, histGPU, histogramsSize*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) cout << "erro: memcpy hists_out " <<  cudaGetErrorString(cudaStatus) << endl;

	cudaStatus = cudaMemcpy(diss_out, dissGPU, qntFrames*sizeof(float), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) cout << "erro: memcpy diss_out " <<  cudaGetErrorString(cudaStatus) << endl;

	//histograms.resize(qntFrames);
	//dissimilarity.resize(qntFrames);
	time.reset(); time.start();

	int count = frameInicial;
	for(int i=0; i<qntFrames; i++)
	{
		dissimilarity[count] = diss_out[i];
		vector<int> hist_tmp;
		for(int j=0; j<256*3; j++)
		{
			hist_tmp.push_back(hists_out[j+i*(256*3)]);
		}
		histograms[count] = hist_tmp;
		//cout << "calculo posicao " << count <<  " " << dissimilarity[count] << endl;
		count++;
	}

	dissimilarity[0] = 0;

	float *var_cpu_out = new float[qntFrames];

	time.stop(); timeD += time.getTimeSec();

	//c//udaStatus = cudaStreamSynchronize(stream1);
	//if(cudaStatus != cudaSuccess) cout << "erro: stream1 " <<  cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaMemcpyAsync(var_cpu_out, var_gpu_out, qntFrames*sizeof(float), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) cout << "erro: memcpy var_out " <<  cudaGetErrorString(cudaStatus) << endl;

	time.reset(); time.start();

	//variances.reserve(qntFrames);
	count = frameInicial;
	for(int i=0; i<qntFrames; i++)
		variances[count++] = (var_cpu_out[i]);

	variances[0] = 0;



	//cout << "outros free "<< endl;
	cudaFree(var_gpu_out);
	cudaFree(histGPU);
	cudaFree(dissGPU);

	delete[] hists_out;
	delete[] diss_out;
	delete[] var_cpu_out;


	//cudaStreamDestroy(stream1);
	//cudaStreamDestroy(stream2);

	time.stop(); timeD += time.getTimeSec();
	Results *result;
	result = Results::getInstance();
	double oldtime = result->getSegmentationWithoutGpuCopy();
	if(oldtime == -1)
		oldtime = 0.0;
	result->setSegmentationWithoutGpuCopy(oldtime + timeD);
	//cout << "fim compute hist var " << frameInicial << endl;


}

extern "C"
void callComputations(string framesDir, vector<double>& variances, vector<double>& dissimilarity, vector<vector<int> >& histograms)
{
	vector<string> frames = FileOperations::listFiles(framesDir, ".jpg");

	histograms.resize(frames.size());
	dissimilarity.resize(frames.size());
	variances.resize(frames.size());

	size_t totalSize = 0;

	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);

	cout << "Memoria CUDA disponivel: " << freeMem/1048576 << "MB" << endl;


	Mat ref = cv::imread(frames[0]);
	int rows = ref.rows;
	int cols = ref.cols;

	int cont = 0;
	int frameInicial = 0;
	while(cont < frames.size())
	{
		cv::Mat concat;
		int framesQnt = 0;
		while(totalSize < freeMem && cont < frames.size() && frames.size() < 50)
		{
			cout << framesQnt << endl;
			if(totalSize + (size_t)((ref.total()*ref.elemSize()) + (256*3*sizeof(int)) + (256*3*sizeof(float))) > freeMem)
				break;

			framesQnt++;
			cv::Mat img = cv::imread(frames[cont]);
			concat.push_back(img);

			totalSize += (size_t)((ref.total()*ref.elemSize()) + (256*3*sizeof(int)) + (256*3*sizeof(float))); //image, histogram, variances size
			cont++;
		}

		computeHistVarDiss(concat.data, frameInicial, framesQnt, rows, cols, variances, dissimilarity, histograms);
		//cout << cont << endl;
		//cout << "sleeping" << endl;
		//sleep(90);

		frameInicial = cont-1;
		totalSize = 0;
		//exit(1);
	}
	//cout << " teste0";
}


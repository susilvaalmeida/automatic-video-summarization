#include "HistogramGPU.cuh"
#include "Defines.h"
#include "Results.h"
#include <iostream>

#define MIN3(x,y,z)  ((y) <= (z) ? \
		((x) <= (y) ? (x) : (y)) \
		: \
		  ((x) <= (z) ? (x) : (z)))
#define MAX3(x,y,z)  ((y) >= (z) ? \
		((x) >= (y) ? (x) : (y)) \
		: \
		  ((x) >= (z) ? (x) : (z)))

__global__ void histograma_kernel_basic(unsigned char *img, int linhas, int colunas, int n, unsigned int *histo)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	int id = (yIndex + xIndex * colunas) + (linhas * colunas * zIndex) ;
	id = id*3;

	if(xIndex < linhas && yIndex < colunas && zIndex < n)
	{
		unsigned char r=img[id+2], g=img[id+1], b=img[id];
		atomicAdd(&(histo[r+(256*3*zIndex)]),1);
		atomicAdd(&(histo[g+(256*3*zIndex)+256]),1);
		atomicAdd(&(histo[b+(256*3*zIndex)+2*256]),1);
	}
}

__global__ void variance_kernel(unsigned int* histo, float* norm, int freq, int n)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	int id = (xIndex + yIndex * 256 * 3) ;

	if(xIndex < 256 && yIndex < n)
	{
		float aux = (float)histo[id]/freq;
		norm[id] = (aux-(1/16))*(aux-(1/16));
	}
}

__global__ void hsv_kernel(unsigned char *img, int linhas, int colunas, int n)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	int id = (yIndex + xIndex * colunas) + (linhas * colunas * zIndex) ;
	id = id*3;

	if(xIndex < linhas && yIndex < colunas && zIndex < n)
	{
		unsigned char h,s,v;
		unsigned char r=img[id+2], g=img[id+1], b=img[id];
		unsigned char rgb_min=MIN3(r,g,b);
		unsigned char rgb_max=MAX3(r,g,b);

		unsigned char delMax = rgb_max - rgb_min;
		h = 0;
		s = 0;
		v = rgb_max;

		if(delMax == 0)
		{
			h = 0; s = 0;
		}
		else
		{
			s = delMax/255;
			if(rgb_max == r)
				h = ((g-b)/delMax)*60;
			else if(rgb_max == g)
				h = (2 + (b-r)/delMax)*60;
			else
				h = (4 + (r-g)/delMax)*60;
		}
		img[id] = h;
		img[id+1] = s;
		img[id+2] = v;
	}
}

__global__ void histograma_kernel_basic(unsigned char *img, int linhas, int colunas, unsigned int *histo)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	int id = (yIndex + xIndex * colunas);

	id = id*3;

	if(xIndex < linhas && yIndex < colunas)
	{
		printf("id:%d (%d-%d)\n", id, xIndex, yIndex);
		unsigned char r=img[id];//, g=img[id+1], b=img[id];
		atomicAdd(&(histo[r]),1);
		//atomicAdd(&(histo[g+256]),1);
		//atomicAdd(&(histo[b+2*256]),1);
	}
}

__global__ void continuous_kernel(cv::gpu::DevMem2D_<uchar4> mat, unsigned char *continuos) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const uchar4& in = mat.ptr(y)[x];
	continuos[y*(mat.cols)+x                      ] = in.z;
	continuos[y*(mat.cols)+x+  (mat.cols*mat.rows)] = in.y;
	continuos[y*(mat.cols)+x+(2*mat.cols*mat.rows)] = in.x;
	//printf("[%d-%d] = (%d,%d,%d)\n", x, y, in.z, in.y, in.x);
}


__global__ void histo_kernel( unsigned char *buffer, long size, unsigned int *histo) {

	__shared__  unsigned int temp[256];
	temp[threadIdx.x+0] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	//i = i*4;
	int stride = blockDim.x * gridDim.x;

	while (i < size) {
		atomicAdd( &temp[buffer[i]], 1 );
		i += stride;
	}

	__syncthreads();
	atomicAdd( &(histo[(threadIdx.x+0)]), temp[threadIdx.x+0] );

}


__global__ void variance_kernel(unsigned int* histo, float *norm, int freq){
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i < 256)
	{
		float aux = (float)histo[i]/freq;
		norm[i] = (aux-(1/256))*(aux-(1/256));
		//printf("%d = %f - %f * 2 = %f\n", i, aux, 1/256, (aux-(1/256))*(aux-(1/256)));
	}
}

__global__ void hsv_kernel(unsigned char *data, int linhas, int colunas)
{
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	int id = (yIndex + xIndex * colunas);

	if(xIndex < linhas && yIndex < colunas)
	{
		unsigned char h,s,v;
		unsigned char r=data[id], g=data[id+(linhas*colunas)], b=data[id+2*linhas*colunas];
		unsigned char rgb_min=MIN3(r,g,b);
		unsigned char rgb_max=MAX3(r,g,b);

		unsigned char delMax = rgb_max - rgb_min;
		h = 0;
		s = 0;
		v = rgb_max;

		if(delMax == 0)
		{
			h = 0; s = 0;
		}
		else
		{
			s = delMax/255;
			if(rgb_max == r)
				h = ((g-b)/delMax)*60;
			else if(rgb_max == g)
				h = (2 + (b-r)/delMax)*60;
			else
				h = (4 + (r-g)/delMax)*60;
		}
		data[id] = h;
		data[id+(linhas*colunas)] = s;
		data[id+2*(linhas*colunas)] = v;
	}
}

HistogramGPU::HistogramGPU()
{
	this->bins = 16;
	this->channels = 3;
}

HistogramGPU::HistogramGPU(vector<float> histogram, int bins, int idFrame, int chan, int freq)
: hist(histogram), idFrame(idFrame), bins(bins), channels(chan), freqTotal(freq)
{
}

vector<float> HistogramGPU::getHistogramNorm()
{
	vector<float> norm;
	for(int i=0; i<BINS; i++)
		norm.push_back(getHistPosNorm(i));
	return norm;
}

vector<HistogramGPU> FeaturesGPU::computeAllHist(unsigned char* images, int frameInicial, int qntFrames, int rows, int cols )
{
	double time = 0.0;
	cv::TickMeter timeLocal;

	timeLocal.reset(); timeLocal.start();


	vector<HistogramGPU> hists;

	//cuda computations
	cudaError_t cudaStatus;

	int freqTotal = rows*cols;
	long size = freqTotal*3*qntFrames;
	int histsSize = 256*3*qntFrames;

	//aloca memoria para imagens e copia para device
	unsigned char* imagemGPU;

	unsigned int* histogramaGPU;

	cudaStatus = cudaMalloc((void**)&imagemGPU, size*sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc imagemGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	cudaStatus = cudaMalloc((void**)&histogramaGPU, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc histogramaGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	timeLocal.stop(); time += timeLocal.getTimeSec();

	//aloca memoria histograma

	cudaStatus = cudaMemcpy(imagemGPU, images, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) cerr << "cudaMemcpy imagemGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	cudaStatus = cudaMemset(histogramaGPU, 0, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMemset histogramaGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	timeLocal.reset(); timeLocal.start();

	//size_t freeMem = 0;
	//size_t totalMem = 0;
	//cudaMemGetInfo(&freeMem, &totalMem);
	//cout << "Memoria CUDA disponivel: " << freeMem/1048576 << "MB" << endl;

	//escolhe tamanhos blocos e grid
	dim3 dimBlock(16,16,4);
	int blocksInX = (rows+16-1)/16;
	int blocksInY = (cols+16-1)/16;
	int blocksInZ = (qntFrames+4-1)/4;
	dim3 dimGrid(blocksInX, blocksInY, blocksInZ);


	//chama kernel histograma
	histograma_kernel_basic<<<dimGrid, dimBlock>>>(imagemGPU, rows, cols, qntFrames, histogramaGPU);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;




	//unsigned int* histTeste = (unsigned int*)malloc(histsSize*sizeof(unsigned int));
	//cudaStatus = cudaMemcpy(histTeste, histogramaGPU, histsSize * sizeof(float), cudaMemcpyDeviceToHost);



	//aloca memoria para calcular variancias
	float* varianciasGPU;
	cudaStatus = cudaMalloc((void**)&varianciasGPU, histsSize*sizeof(float));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc varianciasGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;


	timeLocal.stop(); time += timeLocal.getTimeSec();

	cudaStatus = cudaMemset(varianciasGPU, 0, histsSize*sizeof(float));
	if (cudaStatus != cudaSuccess) cerr << "cudaMemset varianciasGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	//freeMem = 0;
	//totalMem = 0;
	//cudaMemGetInfo(&freeMem, &totalMem);
	//cout << "Memoria CUDA disponivel: " << freeMem/1048576 << "MB" << endl;

	timeLocal.reset(); timeLocal.start();
	//chame kernel variancias
	variance_kernel<<<dimGrid, dimBlock>>>(histogramaGPU, varianciasGPU, freqTotal, qntFrames);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;
	timeLocal.stop(); time += timeLocal.getTimeSec();

	//copia variancias para host
	float* varianciasCPU = (float*)malloc(histsSize*sizeof(float));
	cudaStatus = cudaMemcpy(varianciasCPU, varianciasGPU, histsSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) cerr << "cudaMemcpy varianciasGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;


	cudaFree(histogramaGPU);
	cudaFree(varianciasGPU);

	timeLocal.reset(); timeLocal.start();
	//elimina frames inexpressivos
	unsigned int* framesExcluidos = (unsigned int*)malloc(qntFrames*sizeof(unsigned int));

	float* desvios = (float*)malloc(qntFrames*sizeof(float));
	int count = 0;
	for(int i=0; i<qntFrames; i++)
	{
		desvios[i] = 0;
		for(int j=0; j<256*3; j++)
			desvios[i] += varianciasCPU[(i*256*3)+j];
		float d = sqrt(desvios[i]/(16));
		if(d > 0.23)
		{
			framesExcluidos[count] = i;
			count++;
		}
	}

	//kernel RGB to HSV
	hsv_kernel<<<dimGrid, dimBlock>>>(imagemGPU, rows, cols, qntFrames);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;
	timeLocal.stop(); time += timeLocal.getTimeSec();


	//aloca memoria histograma HSV
	unsigned int* histogramaGPUFinal;
	cudaStatus = cudaMalloc((void**)&histogramaGPUFinal, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc histogramaGPUFinal failed " <<  cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaMemset(histogramaGPUFinal, 0, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMemset histogramaGPUFinal failed " <<  cudaGetErrorString(cudaStatus) << endl;;


	//freeMem = 0;
	//totalMem = 0;
	//cudaMemGetInfo(&freeMem, &totalMem);
	//cout << "Memoria CUDA disponivel: " << freeMem/1048576 << "MB" << endl;


	timeLocal.reset(); timeLocal.start();
	//kernel histograma
	histograma_kernel_basic<<<dimGrid, dimBlock>>>(imagemGPU, rows, cols, qntFrames, histogramaGPUFinal);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;
	timeLocal.stop(); time += timeLocal.getTimeSec();


	//copia historgama final para host
	unsigned int* histogramaCPUFinal = (unsigned int*)malloc(histsSize*sizeof(unsigned int));
	cudaStatus = cudaMemcpy(histogramaCPUFinal, histogramaGPUFinal, histsSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) cerr << "cudaMemcpy histogramaGPUFinal failed " <<  cudaGetErrorString(cudaStatus) << endl;

	timeLocal.reset(); timeLocal.start();
	int idInicial = frameInicial;
	for(int i=0; i<qntFrames; i++)
	{
		idInicial++;

		int excluido = 0;
		for(int j=0; j<count; j++)
		{
			if(framesExcluidos[j] == i)
			{
				excluido = 1;
				break;
			}
		}

		if(excluido)
			continue;

		vector<float> hist(BINS,0.0);
		for(int j=0; j<256; j++)
			hist[j/BINS] += (histogramaCPUFinal[(i*256*3)+j]);

		hists.push_back(HistogramGPU(hist, BINS, idInicial, 3, freqTotal));
	}
	timeLocal.stop(); time += timeLocal.getTimeSec();

	Results *result;
	result = Results::getInstance();
	result->setFeatExtractionGpuCopy(time);

	cudaFree(imagemGPU);
	cudaFree(histogramaGPUFinal);

	free(varianciasCPU);
	free(desvios);
	free(histogramaCPUFinal);
	free(framesExcluidos);

	if (cudaStatus != cudaSuccess)
		exit(1);

	return hists;
}




HistogramGPU FeaturesGPU::computeOneHist(cv::gpu::GpuMat img, int idFrame, int rows, int cols)
{
	//cout << "compute one hist" << endl;
	cv::TickMeter timeLocal; double time;


	timeLocal.reset(); timeLocal.start();
	HistogramGPU hist;

	//cuda computations
	cudaError_t cudaStatus;

	int freqTotal = rows*cols;
	long size = freqTotal*3;
	int histsSize = 256;

	//aloca memoria para imagens e copia para device
	unsigned char* imagemGPU;
	cudaStatus = cudaMalloc((void**)&imagemGPU, size*sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc imagemGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;
	//cudaStatus = cudaMemcpy(imagemGPU, image, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) cerr << "cudaMemcpy imagemGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	//aloca memoria histograma
	unsigned int* histogramaGPU;
	cudaStatus = cudaMalloc((void**)&histogramaGPU, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc histogramaGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;


	timeLocal.stop(); time += timeLocal.getTimeSec();

	cudaStatus = cudaMemset(histogramaGPU, 0, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMemset histogramaGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	timeLocal.reset(); timeLocal.start();

	//escolhe tamanhos blocos e grid
	dim3 dimBlock(16,16);
	int blocksInX = (rows+16-1)/16;
	int blocksInY = (cols+16-1)/16;
	dim3 dimGrid(blocksInY, blocksInX);

	continuous_kernel<<<dimGrid,dimBlock>>>(img,imagemGPU);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;

	img.release();

	//chama kernel histograma
	histo_kernel<<<4, 256>>>(imagemGPU, rows*cols, histogramaGPU);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;

	timeLocal.stop(); time += timeLocal.getTimeSec();

	unsigned int* histTeste = (unsigned int*)malloc(histsSize*sizeof(unsigned int));
	cudaStatus = cudaMemcpy(histTeste, histogramaGPU, histsSize * sizeof(float), cudaMemcpyDeviceToHost);

	//cout << "frame: " << idFrame <<  endl;
	//for(int i=0; i<histsSize; i++)
	//	cout << (int)histTeste[i] << " ";
	//cout << endl;


	//exit(1);
	timeLocal.reset(); timeLocal.start();

	//aloca memoria para calcular variancias
	float* varianciasGPU;
	cudaStatus = cudaMalloc((void**)&varianciasGPU, 256*sizeof(float));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc varianciasGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;
	//cudaStatus = cudaMemset(varianciasGPU, 0, 256*sizeof(float));
	//if (cudaStatus != cudaSuccess) cerr << "cudaMemset varianciasGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;


	//chame kernel variancias
	variance_kernel<<<4, 256>>>(histogramaGPU, varianciasGPU, freqTotal);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;

	timeLocal.stop(); time += timeLocal.getTimeSec();

	//copia variancias para host
	float* varianciasCPU = (float*)malloc(256*sizeof(float));
	cudaStatus = cudaMemcpy(varianciasCPU, varianciasGPU, 256 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) cerr << "cudaMemcpy varianciasGPU failed " <<  cudaGetErrorString(cudaStatus) << endl;

	timeLocal.reset(); timeLocal.start();

	cudaFree(histogramaGPU);
	cudaFree(varianciasGPU);

	//elimina frames inexpressivos
	float desvio = 0.0;

	for(int j=0; j<256; j++)
		desvio += varianciasCPU[j];

	float d = sqrt(desvio);
	//cout << "d:" << d << endl;

	if(d > 0.5)
	{
		return hist;
	}


	//kernel RGB to HSV
	hsv_kernel<<<dimGrid, dimBlock>>>(imagemGPU, rows, cols);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;


	//aloca memoria histograma HSV
	unsigned int* histogramaGPUFinal;
	cudaStatus = cudaMalloc((void**)&histogramaGPUFinal, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMalloc histogramaGPUFinal failed " <<  cudaGetErrorString(cudaStatus) << endl;

	timeLocal.stop(); time += timeLocal.getTimeSec();

	cudaStatus = cudaMemset(histogramaGPUFinal, 0, histsSize*sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) cerr << "cudaMemset histogramaGPUFinal failed " <<  cudaGetErrorString(cudaStatus) << endl;;

	timeLocal.reset(); timeLocal.start();


	//kernel histograma
	histo_kernel<<<4, 256>>>(imagemGPU, rows*cols, histogramaGPUFinal);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) cerr << "failed:" << cudaGetErrorString(cudaStatus) << endl;
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) cerr << "cudaSynchronize failed" << endl;

	timeLocal.stop(); time += timeLocal.getTimeSec();

	//copia historgama final para host
	unsigned int* histogramaCPUFinal = (unsigned int*)malloc(histsSize*sizeof(unsigned int));
	cudaStatus = cudaMemcpy(histogramaCPUFinal, histogramaGPUFinal, histsSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) cerr << "cudaMemcpy histogramaGPUFinal failed " <<  cudaGetErrorString(cudaStatus) << endl;

	timeLocal.reset(); timeLocal.start();

	vector<float> histF(BINS,0.0);
	for(int j=0; j<256; j++)
		histF[j/BINS] += (histogramaCPUFinal[j]);

	//for(int i=0; i<histF.size(); i++)
	//	cout << histF[i] << " ";
	//cout << endl;

	hist.setHistogram(histF);
	hist.setBins(BINS);
	hist.setIdFrame(idFrame);
	hist.setFreqTotal(freqTotal);
	hist.setChannels(3);

	//cout << "size: " << hist.getHistogram().size() << endl;
	cudaFree(imagemGPU);
	cudaFree(histogramaGPUFinal);

	free(varianciasCPU);
	free(histogramaCPUFinal);

	timeLocal.stop(); time += timeLocal.getTimeSec();


	Results *result;
	result = Results::getInstance();
	result->setFeatExtractionGpuCopy(time);

	if (cudaStatus != cudaSuccess)
		exit(1);

	return hist;
}


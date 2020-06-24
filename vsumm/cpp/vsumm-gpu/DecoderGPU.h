#pragma once

#include <string>
#include <vector>
#include "HistogramGPU.cuh"

using namespace std;

class DecoderGPU {
public:
	DecoderGPU(){};
	static int saveFrames(string& videoName, string& dirName);
	static int saveFramesGPU(string& videoName, string& dirName);
	static vector<HistogramGPU> saveFramesAndComputeHistGPU(string& videoName, string& dirName);
};


#pragma once

#include <string>

using namespace std;

class DecoderGPU {
public:
	DecoderGPU(){};
	static int saveFramesGPU(string& videoName, string& dirName);
	static int saveFrames(string& videoName, string& dirName);
};


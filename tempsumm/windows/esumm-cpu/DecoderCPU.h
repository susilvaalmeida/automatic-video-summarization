#pragma once

#include <string>

using namespace std;

class DecoderCPU {
public:
	DecoderCPU(){};
	static int saveFrames(string& videoName, string& dirName);
};


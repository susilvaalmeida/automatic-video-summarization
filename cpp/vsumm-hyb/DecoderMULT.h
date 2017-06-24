#pragma once

#include <string>
#include <vector>

using namespace std;

class DecoderMULT {
public:
	DecoderMULT(){};
	static int saveFrames(string& videoName, string& dirName);
	static int saveFramesOMP(string& videoName, string& dirName);

private:
	static void save_func(string& videoName, string& dirName, int start, vector<int> vetor);
};


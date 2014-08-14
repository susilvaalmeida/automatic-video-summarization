#pragma once

#include <string>
#include <vector>

using namespace std;

class DecoderMULT {
public:
	DecoderMULT(){};
	static int saveFramesTHR(string& videoName, string& dirName);
	static int decodeWithouSaveTHR(string& videoName, string& dirName);
	
	static int saveFramesOMP(string& videoName, string& dirName);
	static int decodeWithouSaveOMP(string& videoName, string& dirName);

private:
	static void save_func(string& videoName, string& dirName, int start, vector<int> vetor);
	static void decode_without_save(string& videoName, string& dirName, int start, vector<int> vetor);
};


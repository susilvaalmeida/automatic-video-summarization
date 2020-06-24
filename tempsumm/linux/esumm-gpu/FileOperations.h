#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

class FileOperations {
public:
	FileOperations(){};

	static int createDir(string dirName);
	static int deleteDir(string dirName);

	static vector<string> listFiles(string dirName, string type);

	static vector<vector<float> > readMatFile(string fileNameComplete);
	static void readMatFile(vector<vector<float> >&, string fileNameComplete);
	static vector<vector<float> > readCtrsFile(string fileNameComplete);

	static int createFile(string fileNameComplete, string content);
	static int copyFile(string fileNameComplete, string newDir);
	static int deleteFile(string fileNameComplete);
	static string getSimpleName(string fileNameComplete);

	static vector<string> split(const string &text, char sep);

};

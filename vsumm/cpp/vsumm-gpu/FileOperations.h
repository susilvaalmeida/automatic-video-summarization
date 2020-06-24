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

	static int createFile(string fileNameComplete, string content);
	static int copyFile(string fileNameComplete, string newDir);
	static int deleteFile(string fileNameComplete);
	static string getSimpleName(string fileNameComplete);

	static vector<string> split(const string &text, char sep);

};

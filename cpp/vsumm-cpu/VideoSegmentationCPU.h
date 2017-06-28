#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <string>

using namespace cv;
using namespace std;

class VideoSegmentationCPU {
public:
	VideoSegmentationCPU(string& video): videoName(video){};
	vector<Mat> readKeepFrames();
    int readSaveFrames(string& dirName);
    int readDiscardFrames();

private:
    string videoName;
};


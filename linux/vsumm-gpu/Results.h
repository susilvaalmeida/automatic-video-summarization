#pragma once

#include <stddef.h>
#include <string>
#include <sstream>
#include "FileOperations.h"
#include <iomanip>

using namespace std;



class Results {
public:
	static Results* getInstance();

	std::string to_string(double x)
	{
		std::ostringstream o;
		o << std::fixed;
		o << std::setprecision(4) << x;
		return o.str();
	}

	std::string to_string(int x)
	{
		std::ostringstream o;
		o  << x;
		return o.str();
	}
	const string& getArch() const {
		return arch;
	}

	void setArch(const string& arch) {
		this->arch = arch;
	}

	double getClustering() const {
		return clustering;
	}

	void setClustering(double clustering) {
		this->clustering = clustering;
	}

	double getClusteringGpuCopy() const {
		return clustering_gpu_copy;
	}

	void setClusteringGpuCopy(double clusteringGpuCopy) {
		clustering_gpu_copy = clusteringGpuCopy;
	}

	double getClusteringParallelPart() const {
		return clustering_parallel_part;
	}

	void setClusteringParallelPart(double clusteringParallelPart) {
		clustering_parallel_part = clusteringParallelPart;
	}

	double getDecode() const {
		return decode;
	}

	void setDecode(double decode) {
		this->decode = decode;
	}

	double getDecodeWithoutGpuCopy() const {
		return decode_without_gpu_copy;
	}

	void setDecodeWithoutGpuCopy(double decodeWithoutGpuCopy) {
		decode_without_gpu_copy = decodeWithoutGpuCopy;
	}

	double getDecodeWithoutWrite() const {
		return decode_without_write;
	}

	void setDecodeWithoutWrite(double decodeWithoutWrite) {
		decode_without_write = decodeWithoutWrite;
	}

	double getEliminateSimilar() const {
		return eliminate_similar;
	}

	void setEliminateSimilar(double eliminateSimilar) {
		eliminate_similar = eliminateSimilar;
	}

	double getFeatExtraction() const {
		return feat_extraction;
	}

	void setFeatExtraction(double featExtraction) {
		feat_extraction = featExtraction;
	}

	double getFeatExtractionGpuCopy() const {
		return feat_extraction_gpu_copy;
	}

	void setFeatExtractionGpuCopy(double featExtractionGpuCopy) {
		feat_extraction_gpu_copy = featExtractionGpuCopy;
	}

	double getFeatExtractionParallelPart() const {
		return feat_extraction_parallel_part;
	}

	void setFeatExtractionParallelPart(double featExtractionParallelPart) {
		feat_extraction_parallel_part = featExtractionParallelPart;
	}

	double getKeyframeExtraction() const {
		return keyframe_extraction;
	}

	void setKeyframeExtraction(double keyframeExtraction) {
		keyframe_extraction = keyframeExtraction;
	}

	int getLength() const {
		return length;
	}

	void setLength(int length) {
		this->length = length;
	}

	const string& getResolution() const {
		return resolution;
	}

	void setResolution(const string& resolution) {
		this->resolution = resolution;
	}

	double getTotal() const {
		return total;
	}

	void setTotal(double total) {
		this->total = total;
	}

	const string& getVideoName() const {
		return video_name;
	}

	void setVideoName(const string& videoName) {
		video_name = videoName;
	}


	void save()
	{
		string text = 	arch + ";" + video_name + ";" + to_string(length) + ";" + resolution  + ";" +
				to_string(decode) + ";" + to_string(decode_without_write) + ";" + to_string(decode_without_gpu_copy) + ";" +
				to_string(feat_extraction) + ";" + to_string(feat_extraction_parallel_part) + ";" + to_string(feat_extraction_gpu_copy) + ";" +
				to_string(clustering) + ";" + to_string(clustering_parallel_part) + ";" + to_string(clustering_gpu_copy) + ";" +
				to_string(keyframe_extraction) + ";" +
				to_string(eliminate_similar) + ";" +
				to_string(total) + "\n";

		FileOperations::createFile("results-vsumm-gpu.txt",text);
	}
protected:
	Results();

private:
	Results(Results const&);
	Results& operator=(Results const&);
	static Results* instance;

	string arch;
	string video_name;
	string resolution;
	int length;

	double decode;
	double decode_without_write;
	double decode_without_gpu_copy;

	double feat_extraction;
	double feat_extraction_parallel_part;
	double feat_extraction_gpu_copy;

	double clustering;
	double clustering_parallel_part;
	double clustering_gpu_copy;

	double keyframe_extraction;
	double eliminate_similar;
	double total;


};

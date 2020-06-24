/*
 * Results.h


 *
 *  Created on: Jul 2, 2014
 *      Author: suellen
 */

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

	const string& getDescriptor() const {
		return descriptor;
	}

	void setDescriptor(const string& descriptor) {
		this->descriptor = descriptor;
	}

	double getFeatDetecDesc() const {
		return feat_detec_desc;
	}

	void setFeatDetecDesc(double featDetecDesc) {
		feat_detec_desc = featDetecDesc;
	}

	double getFeatDetecDescParallelPart() const {
		return feat_detec_desc_parallel_part;
	}

	void setFeatDetecDescParallelPart(double featDetecDescParallelPart) {
		feat_detec_desc_parallel_part = featDetecDescParallelPart;
	}

	double getFeatDetecDescWithoutGpuCopy() const {
		return feat_detec_desc_without_gpu_copy;
	}

	void setFeatDetecDescWithoutGpuCopy(double featDetecDescWithoutGpuCopy) {
		feat_detec_desc_without_gpu_copy = featDetecDescWithoutGpuCopy;
	}

	double getFilter() const {
		return filter;
	}

	void setFilter(double filter) {
		this->filter = filter;
	}

	double getHistogramWords() const {
		return histogram_words;
	}

	void setHistogramWords(double histogramWords) {
		histogram_words = histogramWords;
	}

	double getHistogramWordsParallelPart() const {
		return histogram_words_parallel_part;
	}

	void setHistogramWordsParallelPart(double histogramWordsParallelPart) {
		histogram_words_parallel_part = histogramWordsParallelPart;
	}

	double getHistogramWordsWithoutGpuCopy() const {
		return histogram_words_without_gpu_copy;
	}

	void setHistogramWordsWithoutGpuCopy(double histogramWordsWithoutGpuCopy) {
		histogram_words_without_gpu_copy = histogramWordsWithoutGpuCopy;
	}

	double getLbgClustering() const {
		return lbg_clustering;
	}

	void setLbgClustering(double lbgClustering) {
		lbg_clustering = lbgClustering;
	}

	double getLbgClusteringParallelPart() const {
		return lbg_clustering_parallel_part;
	}

	void setLbgClusteringParallelPart(double lbgClusteringParallelPart) {
		lbg_clustering_parallel_part = lbgClusteringParallelPart;
	}

	double getLbgClusteringWithoutGpuCopy() const {
		return lbg_clustering_without_gpu_copy;
	}

	void setLbgClusteringWithoutGpuCopy(double lbgClusteringWithoutGpuCopy) {
		lbg_clustering_without_gpu_copy = lbgClusteringWithoutGpuCopy;
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

	double getSegmentation() const {
		return segmentation;
	}

	void setSegmentation(double segmentation) {
		this->segmentation = segmentation;
	}

	double getSegmentationParallelPart() const {
		return segmentation_parallel_part;
	}

	void setSegmentationParallelPart(double segmentationParallelPart) {
		segmentation_parallel_part = segmentationParallelPart;
	}

	double getSegmentationWithoutGpuCopy() const {
		return segmentation_without_gpu_copy;
	}

	void setSegmentationWithoutGpuCopy(double segmentationWithoutGpuCopy) {
		segmentation_without_gpu_copy = segmentationWithoutGpuCopy;
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

	double getWordsClutering() const {
		return words_clutering;
	}

	void setWordsClutering(double wordsClutering) {
		words_clutering = wordsClutering;
	}

	void save()
		{
			string text = 	arch + ";" + video_name + ";" + to_string(length) + ";" + resolution  + ";" + descriptor + ";" +
							to_string(decode) + ";" + to_string(decode_without_write) + ";" + to_string(decode_without_gpu_copy) + ";" +
							to_string(segmentation) + ";" + to_string(segmentation_parallel_part) + ";" + to_string(segmentation_without_gpu_copy) + ";" +
							to_string(feat_detec_desc) + ";" + to_string(feat_detec_desc_parallel_part) + ";" + to_string(feat_detec_desc_without_gpu_copy) + ";" +
							to_string(lbg_clustering) + ";" + to_string(lbg_clustering_parallel_part) + ";" + to_string(lbg_clustering_without_gpu_copy) + ";" +
							to_string(histogram_words) + ";" + to_string(histogram_words_parallel_part) + ";" + to_string(histogram_words_without_gpu_copy) + ";" +
							to_string(words_clutering) + ";" +
							to_string(filter) + ";" +
							to_string(total) + "\n";

			FileOperations::createFile("results-esumm-gpu.txt",text);
		}

	const string& getDecodingType() const {
		return decodingType;
	}

	void setDecodingType(const string& decodingType) {
		this->decodingType = decodingType;
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
	string descriptor;
	int length;

	double decode;
	double decode_without_write;
	double decode_without_gpu_copy;

	double segmentation;
	double segmentation_parallel_part;
	double segmentation_without_gpu_copy;

	double feat_detec_desc;
	double feat_detec_desc_parallel_part;
	double feat_detec_desc_without_gpu_copy;

	double lbg_clustering;
	double lbg_clustering_parallel_part;
	double lbg_clustering_without_gpu_copy;

	double histogram_words;
	double histogram_words_parallel_part;
	double histogram_words_without_gpu_copy;

	double words_clutering;

	double filter;

	double total;

	string decodingType;

};

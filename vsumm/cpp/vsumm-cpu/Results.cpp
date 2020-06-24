/*
 * Results.cpp
 *
 *  Created on: Jul 3, 2014
 *      Author: suellen
 */



#include "Results.h"

Results *Results::instance = 0;

Results::Results()
{	
	//string text = 	"arch;video_name;length;resolution;decode;decode_without_write;decode_without_gpu_copy;feat_extraction;feat_extraction_gpu_copy;clustering;keyframe_extraction;eliminate_similar;total";

	//string filename = "results-vsumm-" + arch + ".csv"; 
	//FileOperations::createFile(filename,text);

	length = 0;

	decode = -1;
	decode_without_write = -1;
	decode_without_gpu_copy = -1;

	feat_extraction = -1;
	feat_extraction_gpu_copy = -1;

	clustering = -1;

	keyframe_extraction = -1;
	eliminate_similar = -1;
	total = -1;
}

Results* Results::getInstance()
{
	if(instance == 0)
		instance = new Results();
	return instance;
}

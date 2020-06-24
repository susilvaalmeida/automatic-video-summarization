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
	length = 0;

	decode = -1;
	decode_without_write = -1;
	decode_without_gpu_copy = -1;

	feat_extraction = -1;
	feat_extraction_gpu_copy = -1;

	clustering = -1;
	clustering_parallel_part = -1;
	clustering_gpu_copy = -1;

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

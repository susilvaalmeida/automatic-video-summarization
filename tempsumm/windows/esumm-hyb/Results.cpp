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
	length = -1;

	decode = -1;
	decode_without_write = -1;
	decode_without_gpu_copy = -1;

	segmentation = -1;
	segmentation_parallel_part = -1;
	segmentation_without_gpu_copy = -1;

	feat_detec_desc = -1;
	feat_detec_desc_parallel_part = -1;
	feat_detec_desc_without_gpu_copy = -1;

	lbg_clustering = -1;
	lbg_clustering_parallel_part = -1;
	lbg_clustering_without_gpu_copy = -1;

	histogram_words = -1;
	histogram_words_parallel_part = -1;
	histogram_words_without_gpu_copy = -1;

	words_clutering = -1;

	filter = -1;

	total = -1;

}

Results* Results::getInstance()
{
	if(instance == 0)
		instance = new Results();
	return instance;
}

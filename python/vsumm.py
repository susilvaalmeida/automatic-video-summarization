import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
import video_segmentation as vs
import feature_extraction as feat
import clusterization as cl
import shutil
import time

def vsumm_frames_in_memory(video):
    segmentation = vs.VideoSegmentation(video)
    frames = segmentation.read_and_keep_frames()

    if len(frames) == 0:
        return False

    features = feat.extract_features(frames)
    keyframes = cl.find_clusters(features)

    summary_folder = 'summaryM-'+video[7:-4]
    if not os.path.isdir(summary_folder):
        os.mkdir(summary_folder)

    for k in keyframes:
        frame = frames[k.frame_id-1]
        frame_name = summary_folder+'/frame-'+str(k.frame_id).zfill(6)+'.jpg'
        cv2.imwrite(frame_name,frame)

    return True

def vsumm_frames_in_disk(video):
    frames_folder = 'frames-'+video[7:-4]
    if not os.path.isdir(frames_folder):
        os.mkdir(frames_folder)

    segmentation = vs.VideoSegmentation(video)
    segmentation.read_and_save_frames(frames_folder)
    frames_list = os.listdir(frames_folder)
    features = feat.read_frames_extract_features(frames_folder,frames_list)
    keyframes = cl.find_clusters(features)

    summary_folder = 'summaryD-'+video[7:-4]
    if not os.path.isdir(summary_folder):
        os.mkdir(summary_folder)

    for k in keyframes:
        kframe = frames_folder+'/frame-'+str(k.frame_id).zfill(6)+'.jpg'
        shutil.copy(kframe,summary_folder)


def main():
    videos_list = os.listdir('videos')
    for video in videos_list:
        start = time.time()
        if not vsumm_frames_in_memory('videos/'+video):
            print 'cannot keep all frames in memory'
        end = time.time()
        elapsed_time = end-start
        print 'elapsed time vsumm with frames in memory:', elapsed_time

        start = time.time()
        vsumm_frames_in_disk('videos/'+video)
        end = time.time()
        elapsed_time = end-start
        print 'elapsed time vsumm with frames in disk:', elapsed_time

if __name__ == '__main__':
    main()
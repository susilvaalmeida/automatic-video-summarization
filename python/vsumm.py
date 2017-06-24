import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math
import frame_extraction
import feature_extraction
import clusterization as cl
import shutil

def dist(hist_a, hist_b):
    result = 0.0;
    for a,b in zip(hist_a,hist_b):
        result += np.sqrt((a - b) ** 2);
    return result

def main():
    videos_list = os.listdir('videos')
    for video in videos_list:
        frames_folder = 'frames-'+video[:-4]
        if not os.path.isdir(frames_folder):
            os.mkdir(frames_folder)
        
        frames = frame_extraction.read_frames('videos/'+video)
        #if len(frames) == 0:
        frame_extraction.read_and_save_frames('videos/'+video, frames_folder)
        
        features = feature_extraction.extract_features(frames)
        keyframes = cl.find_clusters(features)

        summary_folder = 'summary-'+video[:-4]
        if not os.path.isdir(summary_folder):
            os.mkdir(summary_folder)

        for k in keyframes:
            kframe = frames_folder+'/frame-'+str(k.frame_id).zfill(6)+'.jpg'
            shutil.copy(kframe,summary_folder)
    

if __name__ == '__main__':
    main()
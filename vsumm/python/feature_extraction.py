import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math

class Histogram:
    def __init__(self, hist, histF, id):
        self.hist = hist
        self.histF = histF
        self.frame_id = id

def compute_histogram(frame, bins=16):
    rgb_hist = []
    color = ('r','g','b')
    for i,col in zip(reversed(range(3)),color):
        hist = cv2.calcHist([frame],[i],None,[bins],[0,256])
        rgb_hist += list(np.array(hist).flatten())
    return rgb_hist

def normalize_hist(histogram):
    frame_size = sum(histogram)/3
    hist_norm = [x/frame_size for x in histogram]
    return hist_norm

def std_dev(histogram):
    hist = normalize_hist(histogram)
    mean,stddev = cv2.meanStdDev(np.array(hist))
    return stddev

def compute_hsv_histogram(frame, bins=16):
    hist_hsv = []
    dev = std_dev(compute_histogram(frame))
    if dev < 0.23:
        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hist_hsv = compute_histogram(frame_hsv)
    return hist_hsv

def extract_features(frames, bins=16):
    hists = []
    for i,frame in enumerate(frames):
        hist_hsv = compute_hsv_histogram(frame)
        if len(hist_hsv) > 0:
            freq = sum(hist_hsv)/3
            hist_norm = [x/freq for x in hist_hsv]
            hist_norm = hist_norm[:bins]
            hists.append(Histogram(hist_norm,hist_hsv[:bins], i+1))
    return hists

def read_frames_extract_features(frames_folder,frames_list, bins=16):
    hists = []
    for i,frame_name in enumerate(frames_list):
        frame = cv2.imread(frames_folder+'/'+frame_name)
        hist_hsv = compute_hsv_histogram(frame)
        if len(hist_hsv) > 0:
            freq = sum(hist_hsv)/3
            hist_norm = [x/freq for x in hist_hsv]
            hist_norm = hist_norm[:bins]
            hists.append(Histogram(hist_norm,hist_hsv[:bins], i+1))
    return hists

import numpy as np
from matplotlib import pyplot as plt
import cv2
import psutil

def read_frames(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print 'cannot read video'
        return []

    fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    mem = psutil.virtual_memory()
    all_frames = []
    
    frames_read = 1
    next_frame = 0
    while frames_read < int(frame_count/fps)+1:
        flag, frame = cap.read()
        
        h,w,c = frame.shape
        if ((h*w*c) * int(frame_count/fps)) > mem.available:
            print 'no memory available to keep the frames'
            return []
        
        if not flag:
            print 'cannot read frame'
            return []

        all_frames.append(frame)
        frames_read += 1
        next_frame += fps
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, next_frame)

    cap.release()
    return all_frames

def read_and_save_frames(video,out_folder):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print 'cannot read video'
        return

    fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    
    frames_saved = 1
    next_frame = 0
    while frames_saved < int(frame_count/fps)+1:
        flag, frame = cap.read()

        if not flag:
            print 'cannot read frame'
            return

        frame_name = out_folder + '/frame-'+str(frames_saved).zfill(6)+'.jpg'

        cv2.imwrite(frame_name, frame)
        frames_saved += 1

        next_frame += fps
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, next_frame)

    cap.release()


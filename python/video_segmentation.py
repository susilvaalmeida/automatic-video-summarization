import cv2
import psutil

class VideoSegmentation:
    def __init__(self, video_name):
        self.video_name = video_name

    def read_and_keep_frames(self):
        cap = cv2.VideoCapture(self.video_name)
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
            if not flag:
                print 'cannot read frame'
                return []

            h,w,c = frame.shape
            if ((h*w*c) * int(frame_count/fps)) > mem.available:
                print 'no memory available to keep the frames'
                return []

            all_frames.append(frame)
            frames_read += 1
            next_frame += fps
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, next_frame)

        cap.release()
        return all_frames

    def read_and_save_frames(self,out_folder):
        cap = cv2.VideoCapture(self.video_name)
        if not cap.isOpened():
            print 'cannot read video'
            return 0

        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        
        frames_saved = 1
        next_frame = 0
        while frames_saved < int(frame_count/fps)+1:
            flag, frame = cap.read()

            if not flag:
                print 'cannot read frame'
                return 0

            frame_name = out_folder+'/frame-'+str(frames_saved).zfill(6)+'.jpg'

            cv2.imwrite(frame_name, frame)
            frames_saved += 1

            next_frame += fps
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, next_frame)

        cap.release()
        return 1

    def read_and_discard_frames(self):
        cap = cv2.VideoCapture(self.video_name)
        if not cap.isOpened():
            print 'cannot read video'
            return 0

        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        
        frames_read = 1
        next_frame = 0
        while frames_read < int(frame_count/fps)+1:
            flag, frame = cap.read()
            if not flag:
                print 'cannot read frame'
                return 0

            frames_read += 1
            next_frame += fps
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, next_frame)

        cap.release()
        return 1
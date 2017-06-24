BIN := vsumm-cpu

CC	  = g++ -std=c++11 -O3

#LIBS      = -L/home/menotti/suellen/libs/opencv-2.4.9/build/lib /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_calib3d.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_contrib.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_core.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_features2d.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_flann.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_gpu.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_highgui.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_imgproc.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_legacy.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_ml.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_nonfree.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_objdetect.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_ocl.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_photo.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_stitching.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_superres.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_ts.a /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_video.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_videostab.so /usr/local/cuda/lib64/libcufft.so /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libnpps.so /usr/local/cuda/lib64/libnppi.so /usr/local/cuda/lib64/libnppc.so /usr/local/cuda/lib64/libcudart.so -lrt -lpthread -lm -ldl
#INCLUDES  = -I/home/menotti/suellen/libs/opencv-2.4.9/include

CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

CPP_SRCS  :=  $(wildcard *.cpp)
HEADERS   := $(wildcard *.h)
CPP_OBJS  := $(patsubst %.cpp, %.o, $(CPP_SRCS))

%.o: %.cpp
	$(CC) $(INCLUDES) -c $< -o $@

$(BIN): $(CPP_OBJS)
	$(CC) -o $(BIN) $(CPP_OBJS) $(LIBS) $(CFLAGS)
clean:
	\rm *.o $(BIN)

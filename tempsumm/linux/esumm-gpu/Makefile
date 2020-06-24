BIN := esumm-gpu

CC	  = g++-4.8 -std=c++11 -O3

CUDA_INSTALL_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCCFLAGS := -arch=sm_30 -use_fast_math

LIBS      = -L/home/menotti/suellen/libs/opencv-2.4.9/build/lib /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_calib3d.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_contrib.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_core.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_features2d.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_flann.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_gpu.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_highgui.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_imgproc.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_legacy.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_ml.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_nonfree.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_objdetect.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_ocl.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_photo.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_stitching.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_superres.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_ts.a /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_video.so /home/menotti/suellen/libs/opencv-2.4.9/build/lib/libopencv_videostab.so /usr/local/cuda/lib64/libcufft.so /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libnpps.so /usr/local/cuda/lib64/libnppi.so /usr/local/cuda/lib64/libnppc.so /usr/local/cuda/lib64/libcudart.so -lrt -lpthread -lm -ldl
INCLUDES  = -I/home/menotti/suellen/libs/opencv-2.4.9/include -I"$(CUDA_INSTALL_PATH)/include"

CPP_SRCS  := $(wildcard *.cpp)
CU_SRCS   := $(wildcard *.cu)
HEADERS   := $(wildcard *.h)
CPP_OBJS  := $(patsubst %.cpp, %.o, $(CPP_SRCS))
CU_OBJS  := $(patsubst %.cu, %.cu_o, $(CU_SRCS))

%.cu_o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(CC) $(INCLUDES) -c $< -o $@

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CC) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS) $(CFLAGS)
clean:
	\rm *.o *.cu_o $(BIN)

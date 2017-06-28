BIN := vsumm-cpu

CC	  = g++ -std=c++11 -O3

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

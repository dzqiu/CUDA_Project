CUDA_PATH 		:= /usr/local/cuda-8.0
TARGET_SIZE 	:= 64
HOST_COMPLIER 	:= g++
NVCC			:= $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPLIER)

NVCCFLAG		:= -m$(TARGET_SIZE)

SMS ?= 30 35 37 50 52 60 
GENCODE_FLAGS :=
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))


INCLUDE := -I /usr/local/include -I /usr/local/include/opencv -I /usr/local/include/opencv2
#INCLUDE += ${shell pkg-config  --libs opencv}



LIBS	:=
LIBS	+= -lopencv_core -lopencv_highgui -lopencv_imgproc
#LIBS	+= -L/usr/local/Qt5.8.0/5.8/gcc_64/lib

CU_FILE := histogram.cu 
TARGET 	:= histogram
all:
	$(NVCC) $(GENCODE_FLAGS) $(NVCCFLAG) $(INCLUDE) $(LIBS) -o $(TARGET)  $(CU_FILE)

run:all 

	./$(TARGET)


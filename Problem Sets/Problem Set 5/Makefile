NVCC=nvcc



OPENCV_LIBPATH=/usr/lib64
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

# or if using MacPorts

#OPENCV_LIBPATH=/opt/local/lib
#OPENCV_INCLUDEPATH=/opt/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH=/opt/cuda/include
CUDA_LIBPATH=/opt/cuda/lib64
CUDA_LIBDEVICPATH=/opt/cuda/nvvm/libdevice --dont-use-profile

#-I $(CUDA_INCLUDEPATH) -ldir $(CUDA_LIBDEVICPATH) -L $(CUDA_LIBPATH)
######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-O3 -arch=sm_20  --ptxas-options=-v -Xcompiler -Wall -Xcompiler -Wextra -m64 

GCC_OPTS=-O3 -Wall -Wextra -m64

histo: main.cu reference_calc.o student.o Makefile
	nvcc -o HW5 main.cu reference_calc.o student.o $(NVCC_OPTS)

student.o: student.cu
	nvcc -c student.cu $(NVCC_OPTS)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o hw *.bin

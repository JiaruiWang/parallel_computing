NVCC=nvcc
###################################
# These are the default install   #
# locations on most linux distros #
###################################


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
######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64



student: main.o student_func.o HW4.o loadSaveImage.o compare.o reference_calc.o Makefile
	$(NVCC) -o HW4 main.o student_func.o HW4.o loadSaveImage.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)  -ldir $(CUDA_LIBDEVICPATH) -L $(CUDA_LIBPATH)

main.o: main.cpp timer.h utils.h reference_calc.h
	g++ -c main.cpp $(GCC_OPTS) 

HW4.o: HW4.cu loadSaveImage.h utils.h
	$(NVCC) -c HW4.cu -I $(OPENCV_INCLUDEPATH) $(NVCC_OPTS)  -ldir $(CUDA_LIBDEVICPATH) -L $(CUDA_LIBPATH)

loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	g++ -c loadSaveImage.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) 

compare.o: compare.cpp compare.h
	g++ -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) 

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) 

student_func.o: student_func.cu reference_calc.cpp utils.h
	$(NVCC) -c student_func.cu $(NVCC_OPTS)  -ldir $(CUDA_LIBDEVICPATH) -L $(CUDA_LIBPATH)

clean:
	rm -f *.o *.png hw

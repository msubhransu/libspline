# This Makefile is used under Linux

MATLABDIR ?= /Applications/MATLAB_R2011a.app
CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -I$(MATLABDIR)/extern/include -I..

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)"
# comment the following line if you use MATLAB on a 32-bit computer
# MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

all: matlab 

matlab: train.$(MEX_EXT) predict.$(MEX_EXT) encode.$(MEX_EXT)

demo:demo.cpp additiveModel.o
	$(CXX) $(CFLAGS) demo.cpp additiveModel.o -o demo

encode.$(MEX_EXT): mexEncode.cpp additiveModel.h additiveModel.o matlabModel.o
	$(MEX) $(MEX_OPTION) mexEncode.cpp additiveModel.o matlabModel.o -o encode.$(MEX_EXT)

predict.$(MEX_EXT): mexPredict.cpp additiveModel.o matlabModel.o
	$(MEX) $(MEX_OPTION) mexPredict.cpp additiveModel.o matlabModel.o -o predict.$(MEX_EXT)

train.$(MEX_EXT): mexTrain.cpp additiveModel.o matlabModel.o
	$(MEX) $(MEX_OPTION) mexTrain.cpp additiveModel.o matlabModel.o -o train.$(MEX_EXT)

matlabModel.o: matlabModel.cpp additiveModel.h
	$(CXX) $(CFLAGS) -c matlabModel.cpp

additiveModel.o: additiveModel.cpp additiveModel.h
	$(CXX) $(CFLAGS) -c additiveModel.cpp

clean:
	rm -f *~ *.o *.mex* *.obj

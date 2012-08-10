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

matlab: splinetrain.$(MEX_EXT) splinepredict.$(MEX_EXT) splineencode.$(MEX_EXT)
#matlab: splinetrain.$(MEX_EXT) splinepredict.$(MEX_EXT)

splineencode.$(MEX_EXT): splineencode.cpp ../splinemodel.h splinemodel.o spline_model_matlab.o
	$(MEX) $(MEX_OPTION) splineencode.cpp splinemodel.o spline_model_matlab.o

splinepredict.$(MEX_EXT): splinepredict.cpp ../splinemodel.h splinemodel.o spline_model_matlab.o
	$(MEX) $(MEX_OPTION) splinepredict.cpp splinemodel.o spline_model_matlab.o

splinetrain.$(MEX_EXT): splinetrain.cpp ../splinemodel.h splinemodel.o spline_model_matlab.o
	$(MEX) $(MEX_OPTION) splinetrain.cpp splinemodel.o spline_model_matlab.o

spline_model_matlab.o: spline_model_matlab.cpp ../splinemodel.h
	$(CXX) $(CFLAGS) -c spline_model_matlab.cpp

splinemodel.o: ../splinemodel.cpp ../splinemodel.h
	$(CXX) $(CFLAGS) -c ../splinemodel.cpp

clean:
	rm -f *~ *.o *.mex* *.obj

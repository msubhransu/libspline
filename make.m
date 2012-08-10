% This make.m is used under Windows
mex -O -largeArrayDims -c ..\splinemodel.cpp
mex -O -largeArrayDims -c spline_model_matlab.c -I..\
mex -O -largeArrayDims splinetrain.c -I..\ splinemodel.obj spline_model_matlab.obj 
mex -O -largeArrayDims splinepredict.c -I..\ splinemodel.obj spline_model_matlab.obj

% This make.m is used under Windows

%compile basic files
mex -O -largeArrayDims -c additiveModel.cpp
mex -O -largeArrayDims -c matlabModel.cpp

% compute matlab binaries
mex -O -largeArrayDims mexTrain.cpp additiveModel.obj matlabModel.obj -o train
mex -O -largeArrayDims mexPredict.cpp additiveModel.obj matlabModel.obj -o predict
mex -O -largeArrayDims mexEncode.cpp additiveModel.obj matlabModel.obj -o encode
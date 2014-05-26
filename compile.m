% Entry code for compiling all the mex files
% Use option -std=c++11 for clang compiler on OSX 10.9
% Use option -largeArrayDims for 64 bit machines
%
% Author: Subhransu Maji

% Mex code for rectangle parsing
mex -O -largeArrayDims CXXFLAGS="-std=c++11" mexTrain.cpp additiveModel.cpp matlabModel.cpp -o train
disp('done compiling..train.');

% Mex code for tiered parsing
mex -O -largeArrayDims CXXFLAGS="-std=c++11" mexPredict.cpp additiveModel.cpp matlabModel.cpp -o predict
disp('done compiling..predict.');

% Mex code for maxflow
mex -O -largeArrayDims CXXFLAGS="-std=c++11" mexEncode.cpp additiveModel.cpp matlabModel.cpp -o encode
disp('done compiling..encode.');


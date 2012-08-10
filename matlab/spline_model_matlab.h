#ifndef _SPLINE_MODEL_MATLAB_H
#define _SPLINE_MODEL_MATLAB_H 
const char *model_to_matlab_structure(mxArray *plhs[], splineModel *model_);
const char *matlab_matrix_to_model(splineModel *model_, const mxArray *matlab_struct);
#endif
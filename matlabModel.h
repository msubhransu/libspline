#ifndef _ADDITIVE_MODEL_MATLAB_H
#define _ADDITIVE_MODEL_MATLAB_H
const char *model_to_matlab_structure(mxArray *plhs[], additiveModel *model_);
const char *matlab_matrix_to_model(additiveModel *model_, const mxArray *matlab_struct);
#endif

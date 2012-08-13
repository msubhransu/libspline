#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "mex.h"
#include "additiveModel.h"
#include "matlabModel.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

// data
int col_format_flag, nvec, dim;
double **x, *y;
additiveModel *model;

// functions
// help message
void exit_with_help(){
  mexPrintf(
	    "Usage: [predicted_label,accuracy,decision_values] = predict(test_label_vector, test_instance_matrix, model, 'col');\n"
	    "options:\n"
	    "	if 'col' is set, test_instance_matrix is parsed in column format, otherwise is in row format\n"
	    );
}

// parse options in the command line
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name){
  char cmd[CMD_LEN];
  col_format_flag = 0;
  
  if(nrhs <= 1)
    return 1;
  
  if(nrhs == 4){
    mxGetString(prhs[3], cmd, mxGetN(prhs[3])+1);
    if(strcmp(cmd, "col") == 0)
      col_format_flag = 1;
  }
  return 0;
}

// empty output constructor
static void fake_answer(mxArray *plhs[]){
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

// prediction function
int do_predict(mxArray *plhs[]){
  if(model->dim !=dim){
    mexPrintf("Error: model feature dim != feature dim\n");
    return -1;
  }
  plhs[0] = mxCreateDoubleMatrix(nvec,1,mxREAL);
  plhs[1] = mxCreateDoubleMatrix(3,1,mxREAL);
  plhs[2] = mxCreateDoubleMatrix(nvec,1,mxREAL);
  
  double *ptr_labels = mxGetPr(plhs[0]);
  double *ptr_acc   = mxGetPr(plhs[1]);
  double *ptr_dec_values = mxGetPr(plhs[2]);
  
  // compute predictions
  model->predict(x,ptr_dec_values,ptr_labels, nvec); 
  
  // compute accuracy
  int numcorrect;
  double accuracy, prec, recall;
  model->getAccuracy(ptr_dec_values,y,nvec,numcorrect,accuracy,prec,recall);
  
  // assign it to the ouputs
  ptr_acc[0] = accuracy;
  ptr_acc[1] = prec;
  ptr_acc[2] = recall;
	
  return 0;
}

// read the input features into internal data structures
int read_problem_dense(const mxArray *label_vec, const mxArray *instance_mat){
  int i, label_vector_row_num;
  double *samples;
  mxArray *instance_mat_col; // instance sparse matrix in column format
  
  if(col_format_flag)
    instance_mat_col = (mxArray *)instance_mat;
  else{
    // transpose instance matrix
    mxArray *prhs[1], *plhs[1];
    prhs[0] = mxDuplicateArray(instance_mat);
    if(mexCallMATLAB(1, plhs, 1, prhs, "transpose")){
      mexPrintf("Error: cannot transpose training instance matrix\n");
      return -1;
    }
    instance_mat_col = plhs[0];
    mxDestroyArray(prhs[0]);
  }

  // compute the dimensions of the features
  nvec = (int) mxGetN(instance_mat_col);
  dim  = (int) mxGetM(instance_mat_col);
  
  label_vector_row_num = (int) mxGetM(label_vec);
  if(label_vector_row_num!= nvec){
    mexPrintf("Length of label vector does not match # of instances.\n");
    return -1;
  }
	
  // obtain pointers to the data/labels
  y = mxGetPr(label_vec);
  samples = mxGetPr(instance_mat_col);
  x = new double*[nvec];
  
  for(i=0;i<nvec;i++)
    x[i] = &samples[i*dim];
  
  return 0;
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]){
  const char *error_msg=NULL;

  // transform the input matrix to libspline format
  if(nrhs > 0 && nrhs < 5){
    int err=0;
    
    if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
      mexPrintf("Error: label vector and instance matrix must be double\n");
      fake_answer(plhs);
      return;
    }
    
    if(parse_command_line(nrhs, prhs, NULL)){
      exit_with_help();
      fake_answer(plhs);
      return;
    }
		
    if(!mxIsSparse(prhs[1]))
      err = read_problem_dense(prhs[0], prhs[1]);
    else{
      mexPrintf("Error: test_instance_matrix must be dense\n");
      fake_answer(plhs);
      return;
    }
    
    if(err || error_msg){
      if (error_msg != NULL)
	mexPrintf("Error: %s\n", error_msg);
      fake_answer(plhs);
      delete [] x;
      return;
    }
    
		
    //initialize an empty model
    model = new additiveModel();
    
    error_msg = matlab_matrix_to_model(model,prhs[2]);
    
    if(error_msg){
      mexPrintf("Error: can't read model: %s\n", error_msg);
      delete [] x;
      delete model;
      fake_answer(plhs);
      return;
    }

    //compute predictions
    if(do_predict(plhs) < 0){
      delete [] x;
      delete [] model;
      fake_answer(plhs);
      return;
    }
    delete [] x;
    delete model;
  }else{
    exit_with_help();
    fake_answer(plhs);
    return;
  }
}

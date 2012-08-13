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

// variables

parameter param;		// options set by parse_command_line
additiveModel *model;		// spline model 
bool inputModel = false;	// can provide model instead of optionf for encoding.
int col_format_flag, nvec, dim; // other features
double **x;			// pointer to features

// functions

// help message
void exit_with_help(){
  mexPrintf(
	    "Usage: [encodedFeats, model] = encode(feats, [model or 'options'], 'col');\n"
	    "outputs:\n"
	    "encodedFeats	: encoded features (Note, these are in 'col' format) \n"
	    "model		: if options are provided instead of a model, then returns a model\n"		  
	    "\noptions:\n"
	    "-t type		: O: Spline, 1: Trigonometric, 2: Hermite (default=0, t={0,1,2} )\n"
	    "-d degree	: set the degree of B-Spline basis (default=1, d={0,1,2,3} )\n"
	    "-r reg		: set the order of regularization (default=1) r={0,1,2,...}\n"
	    "-n bins		: set the number of bins (default 10)\n"
	    );
}

// encode features
int encode_features(mxArray *plhs[]){
  //matlab matrices are in column format
  plhs[0] = mxCreateDoubleMatrix(model->wdim,nvec, mxREAL);
  double *feat  = mxGetPr(plhs[0]);
  
  int i, j, k, fo, ei;
  double *xi;
  
  //space for encodings
  double *ew = new double[model->degree];
  double *xd = new double[model->numbasis];
  
  fo=0; //feature offset
  double *st = model->st;
  for(i = 0; i < nvec; i++){
    xi = x[i];
    for(j = 0; j < dim ; j++){
      if(st[j] > 0){
	if(model->encoding == SPLINE){
	  model->bSplineEncoder(xi[j],j,ei,ew);
	  if(model->reg == 0){ //identity matrix 
	    for(k=0; k <= model->degree;k++){
	      feat[fo+ei-k] += st[j]*ew[model->degree-k];
	    }
	  }else{ //D_d matrix regularization
	    model->projectDenseW(ei,ew,st[j],xd);
	    for(k=0; k < model->numbasis;k++)
	      feat[fo+k] += xd[k];
	  }
	}else if(model->encoding == TRIGONOMETRIC){
	  model->trigEncoder(xi[j],j,xd);
	  for(k=0; k < model->numbasis;k++)
	    feat[fo+k] += xd[k];
	}else if(model->encoding == HERMITE){
	  model->hermiteEncoder(xi[j],j,xd);
	  for(k=0; k < model->numbasis;k++)
	    feat[fo+k] += xd[k];
	}
      }
      fo += model->numbasis;
    }
  }
  delete [] ew;
  delete [] xd;
  return 0;
}

// parse command line
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name){
  int i, argc = 1;
  char cmd[CMD_LEN];
  char *argv[CMD_LEN/2];
  
  //set default parameters
  param.encoding = SPLINE;
  param.degree = 1;
  param.reg = 1;
  param.Cp = 1;
  param.Cn = 1;
  param.eps = 0.1; 
  param.numbins = 10;
  param.bias = 10;
  
  col_format_flag = 0;
  
  if(nrhs <= 1)
    return 1;
  
  if(nrhs == 3){
    mxGetString(prhs[2], cmd, mxGetN(prhs[2])+1);
    if(strcmp(cmd, "col") == 0)
      col_format_flag = 1;
  }
  
  // put options in argv[]
  if(nrhs > 1){
    mxGetString(prhs[1], cmd,  mxGetN(prhs[1]) + 1);
    if((argv[argc] = strtok(cmd, " ")) != NULL)
      while((argv[++argc] = strtok(NULL, " ")) != NULL)
	;
  }
  
  // parse options
  for(i=1;i<argc;i++){
    if(argv[i][0] != '-') break;
    ++i;
    
    switch(argv[i-1][1]){
    case 't':
      param.encoding = atoi(argv[i]);
      break;
    case 'd':
      param.degree = atoi(argv[i]);
      break;
    case 'r':
      param.reg = atoi(argv[i]);
      break;
    case 'n':
      param.numbins = atoi(argv[i]);
      break;
    default:
      mexPrintf("Error: unknown option\n", argv[i-1][1]);
      return 1;
    }
  }
  return 0;
}

// empty output constructor
static void fake_answer(mxArray *plhs[]){
  plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
  if(!inputModel)
    plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

// read the features into the internal data structures
int read_problem_dense(const mxArray *instance_mat){
  int i;
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
  
  // compute the dimensions of the input features
  nvec = (int) mxGetN(instance_mat_col);
  dim  = (int) mxGetM(instance_mat_col);
  
  samples = mxGetPr(instance_mat_col);
  x = new double*[nvec];
	
  for(i=0;i<nvec;i++)
    x[i] = &samples[i*dim];
  return 0;
}

// check the parameters
const char* check_parameter(const parameter * param){
  if(param->encoding < 0 || param->encoding > 3){
    return "encoding type should be {0,1,2}";
  }

  if(param->encoding == 0 && (param->degree < 0 || param->degree > 3)){
    return "degree should be {0,1,2,3} for splines";
  }
  
  if(param->reg < 0){
    return "regularization should be >= 0";
  }
  
  if(param->numbins < 1){
    return "numbins < 1";
  }
  
  if(param->Cp <= 0){
    return "Cp < 0";
  }
  
  if(param->Cn <= 0){
    return "Cn < 0";
  }
  
  if(param->eps <= 0){
    return "eps <= 0";
  }
  return NULL;
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]){
  const char *error_msg;
  
  if(nrhs > 0 && nrhs < 4){
    int err=0;
    
    if(!mxIsDouble(prhs[0])) {
      mexPrintf("Error: feats must be double.\n");
      fake_answer(plhs);
      return;
    }
    if(!mxIsSparse(prhs[0]))
      err = read_problem_dense(prhs[0]);
    else{
      mexPrintf("Error: fesats must be dense.\n");
      fake_answer(plhs);
      return;
    }
    if (nrhs > 1){
      if(mxIsClass(prhs[1],"char")){ // options are specified
	inputModel = false;
	if(parse_command_line(nrhs, prhs, NULL)){
	  exit_with_help();
	  fake_answer(plhs);
	  delete [] x;
	  return;
	}
	model = new additiveModel(&param, x, dim, nvec); //initialize
      }else{
	inputModel = true;
	model = new additiveModel();
	error_msg = matlab_matrix_to_model(model,prhs[1]);
	if(error_msg){
	  mexPrintf("Error: can't read model: %s\n", error_msg);
	  delete [] x;
	  delete model;
	  fake_answer(plhs);
	  return;
	}
      }
    }else{ //use default params
      inputModel = false;
      parse_command_line(nrhs, prhs, NULL);
      model = new additiveModel(&param, x, dim, nvec);
    }
    error_msg = check_parameter(&param);
    
    if(err || error_msg){
      if (error_msg != NULL)
	mexPrintf("Error: %s\n", error_msg);
      delete [] x;
      delete model;
      fake_answer(plhs);
      return;
    }
    err = encode_features(plhs);
    
    const char *error_msg;
    if(!inputModel){
      //return the precompute model if not provided as input
      error_msg = model_to_matlab_structure(plhs+1, model);
      if(error_msg)
	mexPrintf("Error: can't convert spline model to matrix structure: %s\n", error_msg);
    }
    delete [] x;
    delete model;
  }else{
    exit_with_help();
    fake_answer(plhs);
    return;
  }
}

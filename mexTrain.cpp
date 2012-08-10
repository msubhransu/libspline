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

parameter param;				// parameters set by parse_command_line
additiveModel *model;				// spline model
int col_format_flag, nvec, dim;	// other options
double **x, *y;					// input training data/labels


void print_null(const char *s){}

void exit_with_help()
{
	mexPrintf(
	"Usage: model = train(training_label_vector, training_instance_matrix, 'options', 'col');\n"
	"options:\n"
	"-t type     : 0: Spline, 1: Trigonometric, 2: Hermite  (default=0)\n"
	"-d degree   : set the B-Spline degree (default=1) d={0,1,2,3}\n"
	"-r reg      : set the order of regularization (default=1) r={0,1,2,...}\n"
	"-n bins     : set the number of bins (default 10)\n"
	"-c cost     : set the parameter C (default 1)\n"
	"-e epsilon  : set tolerance of termination criterion\n"
	"		       Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"-B bias     : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default 1)\n"
	"-wi weight  : weights adjust the parameter C of different classes (see README for details)\n"
	"col:\n"
	"	if 'col' is set, training_instance_matrix is parsed in column format, otherwise is in row format\n"
	);
}


// nrhs should be 3
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
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
	
	//per class weights
	int nr_weight=0; 
	
	col_format_flag = 0;
		
	if(nrhs <= 1)
		return 1;

	if(nrhs == 4)
	{
		mxGetString(prhs[3], cmd, mxGetN(prhs[3])+1);
		if(strcmp(cmd, "col") == 0)
			col_format_flag = 1;
	}

	// put options in argv[]
	if(nrhs > 2)
	{
		mxGetString(prhs[2], cmd,  mxGetN(prhs[2]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		
		switch(argv[i-1][1])
		{
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
			case 'c':
				param.Cn = param.Cp = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				param.bias = atof(argv[i]);
				break;
			case 'w':
				++nr_weight;
				if(atoi(&argv[i-1][2]) > 0)
					param.Cp *= atof(argv[i]);
				else
					param.Cn *= atof(argv[i]);
				break;
			default:
				mexPrintf("unknown option\n");
				return 1;
		}
	}
	return 0;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int read_problem_dense(const mxArray *label_vec, const mxArray *instance_mat)
{
	int i, label_vector_row_num;
	double *samples;
	
	mxArray *instance_mat_col; // instance sparse matrix in column format


	if(col_format_flag)
		instance_mat_col = (mxArray *)instance_mat;
	else
	{
		// transpose instance matrix
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// the number of instance
	nvec = (int) mxGetN(instance_mat_col);
	dim  = (int) mxGetM(instance_mat_col);
	
	label_vector_row_num = (int) mxGetM(label_vec);

	if(label_vector_row_num!= nvec)
	{
		mexPrintf("Error: Length of label vector does not match # of instances.\n");
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
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;

	// Transform the input Matrix to libsvm format
	if(nrhs > 0 && nrhs < 5)
	{
		int err=0;

		if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
			mexPrintf("Error: label vector and instance matrix must be double\n");
			fake_answer(plhs);
			return;
		}

		if(parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			fake_answer(plhs);
			return;
		}
		
		if(!mxIsSparse(prhs[1]))
			err = read_problem_dense(prhs[0], prhs[1]);
		else
		{
			mexPrintf("Error : training_instance_matrix must be dense\n");
			fake_answer(plhs);
			return;
		}

		error_msg = check_parameter(&param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				mexPrintf("Error: %s\n", error_msg);
			fake_answer(plhs);
			delete [] x;
			return;
		}

		const char *error_msg;
		model = new additiveModel(&param, x, dim, nvec); //initialize
		model->train(x,y,nvec,&param); //train the model
		error_msg = model_to_matlab_structure(plhs, model);

		if(error_msg)
			mexPrintf("Error: can't convert spline model to matrix structure: %s\n", error_msg);
		delete model;
		delete [] x;
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}

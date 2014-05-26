/* Author : Subhransu Maji
 *
 * Code for encoding the matlab model
 *
 * Version 1.0
 */
#include <stdlib.h>
#include <string.h>
#include "additiveModel.h"
#include "mex.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define NUM_OF_RETURN_FIELD 17

static const char *field_names[] = {
	"encoding", 
	"degree",
	"regularization",
	"numbins",
	"numbasis",
	"dim",
	"wdim",
	"xmin",
	"xmax",
	"xmean",
	"xvar",
	"st",
	"a",
	"b",
	"dimwts",
	"w",
	"bias",
};

// convert model to matlab
const char *model_to_matlab_structure(mxArray *plhs[], additiveModel *model_){
  int i,out_id=0;
  double *ptr; 
  mxArray *return_model, **rhs;
  
  rhs = (mxArray **)mxMalloc(sizeof(mxArray *)*NUM_OF_RETURN_FIELD);
  
  //encoding
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->encoding;
  out_id++;
  
  // degree
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->degree;
  out_id++;
  
  // regularization
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->reg;
  out_id++;
  
  // numbins
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->numbins;
  out_id++;
  
  // numbasis
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->numbasis;
  out_id++;
  
  // dim
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->dim;
  out_id++;
  
  // wdim 
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->wdim;
  out_id++;
  
  //xmin
  rhs[out_id] = mxCreateDoubleMatrix(model_->dim, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->dim; i++)
    ptr[i] = model_->xmin[i];
  out_id++;
  
  
  //xmax
  rhs[out_id] = mxCreateDoubleMatrix(model_->dim, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->dim; i++)
    ptr[i] = model_->xmax[i];
  out_id++;
  
  //xmean
  rhs[out_id] = mxCreateDoubleMatrix(model_->dim, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->dim; i++)
    ptr[i] = model_->xmean[i];
  out_id++;
  
  //xvar
  rhs[out_id] = mxCreateDoubleMatrix(model_->dim, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->dim; i++)
    ptr[i] = model_->xvar[i];
  out_id++;
  
  //st
  rhs[out_id] = mxCreateDoubleMatrix(model_->dim, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->dim; i++)
    ptr[i] = model_->st[i];
  out_id++;
  
  //a
  rhs[out_id] = mxCreateDoubleMatrix(model_->dim, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->dim; i++)
    ptr[i] = model_->a[i];
  out_id++;
  
  //b
  rhs[out_id] = mxCreateDoubleMatrix(model_->dim, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->dim; i++)
    ptr[i] = model_->b[i];
  out_id++;
  
  //dimwts
  if(model_->dimwts != NULL){
    rhs[out_id] = mxCreateDoubleMatrix(model_->numbasis, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    for(i = 0; i < model_->numbasis; i++)
      ptr[i] = model_->dimwts[i];
  }else {
    rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  }
  
  out_id++;
  
  //w
  rhs[out_id] = mxCreateDoubleMatrix(1, model_->wdim, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  for(i = 0; i < model_->wdim; i++)
    ptr[i]=model_->w[i];
  out_id++;
  
  // bias
  rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
  ptr = mxGetPr(rhs[out_id]);
  ptr[0] = model_->bias;
  out_id++;
  
  // create a struct matrix contains NUM_OF_RETURN_FIELD fields 
  return_model = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);
  
  // fill struct matrix with input arguments 
  for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
    mxSetField(return_model,0,field_names[i],mxDuplicateArray(rhs[i]));
  
  // return model
  plhs[0] = return_model;
  mxFree(rhs);
  
  return NULL;
}

// convert matlab struct to model
const char *matlab_matrix_to_model(additiveModel *model_, const mxArray *matlab_struct){
    int i, num_of_fields;
    double *ptr;
    int id = 0;
    mxArray **rhs;
    
    num_of_fields = mxGetNumberOfFields(matlab_struct);
    rhs = (mxArray **) mxMalloc(sizeof(mxArray *)*num_of_fields);
    
    for(i=0;i<num_of_fields;i++)
      rhs[i] = mxGetFieldByNumber(matlab_struct, 0, i);

    // encoding
    ptr = mxGetPr(rhs[id]);
    model_->encoding = (int)ptr[0];
    id++;
    
    // degree
    ptr = mxGetPr(rhs[id]);
    model_->degree = (int)ptr[0];
    id++;
    
    // regularization
    ptr = mxGetPr(rhs[id]);
    model_->reg = (int)ptr[0];
    id++;
    
    // numbins
    ptr = mxGetPr(rhs[id]);
    model_->numbins = (int)ptr[0];
    id++;

    // numbasis
    ptr = mxGetPr(rhs[id]);
    model_->numbasis = (int)ptr[0];
    id++;
    
    // dim
    ptr = mxGetPr(rhs[id]);
    model_->dim = (int)ptr[0];
    id++;
    
    // wdim
    ptr = mxGetPr(rhs[id]);
    model_->wdim = (int)ptr[0];
    id++;
    
	
    // xmin
    ptr = mxGetPr(rhs[id]);
    model_->xmin = new double[model_->dim];
    for(i=0; i<model_->dim; i++)
      model_->xmin[i]=ptr[i];
    id++;
    
    // xmax
    ptr = mxGetPr(rhs[id]);
    model_->xmax = new double[model_->dim];
    for(i=0; i<model_->dim; i++)
      model_->xmax[i]=ptr[i];
    id++;
    
	
    // xmean
    ptr = mxGetPr(rhs[id]);
    model_->xmean = new double[model_->dim];
    for(i=0; i<model_->dim; i++)
      model_->xmean[i]=ptr[i];
    id++;
	
    // xvar
    ptr = mxGetPr(rhs[id]);
    model_->xvar = new double[model_->dim];
    for(i=0; i<model_->dim; i++)
      model_->xvar[i]=ptr[i];
    id++;
	
    // st
    ptr = mxGetPr(rhs[id]);
    model_->st = new double[model_->dim];
    for(i=0; i<model_->dim; i++)
      model_->st[i]=ptr[i];
    id++;

    // a
    ptr = mxGetPr(rhs[id]);
    model_->a = new double[model_->dim];
    for(i=0; i<model_->dim; i++)
      model_->a[i]=ptr[i];
    id++;
	
    // b
    ptr = mxGetPr(rhs[id]);
    model_->b = new double[model_->dim];
    for(i=0; i<model_->dim; i++)
      model_->b[i]=ptr[i];
    id++;
    
    // dimwts
    ptr = mxGetPr(rhs[id]);
    if(model_->encoding != SPLINE){
      model_->dimwts = new double[model_->numbasis];
      for(i=0; i<model_->numbasis; i++)
	model_->dimwts[i]=ptr[i];
    }else {
      model_->dimwts = NULL;
    }
    id++;
    
    // w
    ptr = mxGetPr(rhs[id]);
    model_->w=new double[model_->wdim];
    for(i = 0; i < model_->wdim; i++)
      model_->w[i]=ptr[i];
    id++;
    
    // bias
    ptr = mxGetPr(rhs[id]);
    model_->bias = ptr[0];
    id++;
    
    mxFree(rhs);
    return NULL;
}
  
  

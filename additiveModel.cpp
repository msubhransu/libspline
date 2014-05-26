/* Author : Subhransu Maji
 *
 * Implements encoding methods
 *
 * Version 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "additiveModel.h"

#define INF 1e10

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifndef PI
#define PI 3.14159265358979
#endif

//return the b-spline basis index for a given dimension
void additiveModel::getBasisIndex(double x, 
				  int dimidx, 
				  int &ei, 
				  double &ai){
  double fi = a[dimidx]*x + b[dimidx];
  ei = (int)fi;
  ai = fi-ei;
  if(ei < degree){
    ei=degree; 
    ai=0;
  }else if(ei >= numbasis){ // numbasis = numbins + degree
    ei = numbasis-1;
    ai = 1;
  }
}

//B-Spline embedding
void additiveModel::bSplineEncoder(double x,
				   int dimidx,
				   int &ei,
				   double *wts){
  double t;
  getBasisIndex(x,dimidx,ei,t);
  double t2 = t*t, t3 = t2*t;
  switch(degree){
  case 0:
    wts[0] = 1;
    break;
  case 1:
    wts[0] = 1-t;
    wts[1] = t;
    break;
  case 2:
    wts[0] = 0.5*t2 - t + 0.5;
    wts[1] =    -t2 + t + 0.5;
    wts[2] = 0.5*t2          ;
    break;
  case 3:
    wts[0] = 1.0/6*(1-t)*(1-t)*(1-t);
    wts[1] = 1.0/6*(4 - 6*t2 + 3*t3);
    wts[2] = 1.0/6*(1 + 3*t + 3*t2 - 3*t3);
    wts[3] = 1.0/6*(t3);
    break;
  default:
    ;
    //should not happen	
  }
}

// Trigonometic embedding 
void additiveModel::trigEncoder(double x, int dimidx, double *xd){
  double cx = (xmax[dimidx] + xmin[dimidx])/2;
  double dx = (xmax[dimidx] - xmin[dimidx])/2;
  double t = PI*(x - cx)/dx;
  double sint = sin(t); 
  double cost = cos(t);
  
  xd[0] = sint; xd[1] = cost;
  for(int i = 1; i < degree ; i++){
    xd[2*i] = xd[2*i-2]*cost + xd[2*i-1]*sint; 
    xd[2*i+1] = xd[2*i-1]*cost - xd[2*i-2]*sint;
  }
  for(int i = 0; i < numbasis; i++){
    xd[i] *= st[dimidx]*dimwts[i];
  }
}

// Hermite embedding
void additiveModel::hermiteEncoder(double x, int dimidx, double *xd){
  double cx = (xmax[dimidx] + xmin[dimidx])/2;
  double dx = (xmax[dimidx] - xmin[dimidx])/2;
  double t  = (x - cx)/dx;
  xd[0] = t;
  if(numbasis > 1)
    xd[1] = (t*t-1);
  for(int i = 2; i < numbasis ; i++){
    xd[i] = t*xd[i-1] - i*xd[i-2];
  }
  for(int i = 0; i < numbasis; i++){
    xd[i] *= st[dimidx]*dimwts[i];
  }
}

//empty additiveModel constructor
additiveModel::additiveModel(){
  encoding = SPLINE;
  degree = 1; 
  reg = 1; 
  numbins = 0;
  numbasis = 0;
  dim = 0;
  xmax = NULL;
  xmin = NULL;
  xmean = NULL;
  xvar = NULL;
  st = NULL;
  a = NULL;
  b = NULL;
  w = NULL;
  bias = 0;
}

//initialize a additiveModel given the data
additiveModel::additiveModel(const parameter *param,
			     double **x,
			     int fdim, 
			     int nvec){
  //allocate all the memory for the model
  int i,j;
  encoding = param->encoding;
  degree = param->degree;
  reg = param->reg;
  numbins = param->numbins;
  if(encoding == SPLINE){
    numbasis = numbins + degree;
  }else if(encoding == TRIGONOMETRIC){
    numbasis = degree*2;
  }else if(encoding == HERMITE){
    numbasis = degree;
  }

  dim = fdim;
  wdim = dim*numbasis;
  xmax = new double[dim];
  xmin = new double[dim];
  xmean = new double[dim];
  xvar = new double[dim];
  st = new double[dim];
  a = new double[dim];
  b = new double[dim];
  w = new double[wdim];
  dimwts = NULL;
  if(encoding == TRIGONOMETRIC){
    dimwts = new double[numbasis];
    for(i=0;i < degree;i++){
      dimwts[2*i] = 1.0/pow(i+1,reg);
      dimwts[2*i+1] = dimwts[2*i];
    }
  }else if(encoding == HERMITE){
    dimwts = new double[numbasis];
    double normsq = 1.0;
    dimwts[0] = 1.0;
    if(reg == 1){
      for(i=1; i < numbasis;i++){
	normsq = normsq*(i+1)*(i+1)/i; 
	dimwts[i] = 1./sqrt(normsq);
      }
    }if(reg == 2){
      if(numbasis > 1)
	dimwts[1] = 1.0/2;
			
      normsq = 1.0/2;
      for(i=2; i < numbasis; i++){
	normsq = normsq*(i+1)*(i+1)/(i-1);
	dimwts[i] = 1./sqrt(normsq);
      }
    }
  }
		
  //clear weights
  for(i=0;i<wdim;i++)
    w[i] = 0;
	
  //uniformly sample points in the [min,max] range in each dimension
  double tmpMAX, tmpMIN, step_size, xsum, xsumsq;
  for(i=0;i<dim;i++){
    tmpMAX = -INF;
    tmpMIN = INF;
    xsum = 0; xsumsq = 0;
    for(j=0;j<nvec;j++){
      xsum += x[j][i];
      xsumsq += x[j][i]*x[j][i];
			
      if(x[j][i] < tmpMIN)
	tmpMIN = x[j][i];
      if(x[j][i] > tmpMAX)
	tmpMAX = x[j][i];
    }
    //update min and max 
    xmin[i]  = tmpMIN;
    xmax[i]  = tmpMAX;
    xmean[i] = xsum/nvec;
		
    //update linear interpolation paramters
    if(tmpMAX - tmpMIN > 1e-10){
      step_size = (tmpMAX - tmpMIN)/param->numbins;
      st[i] = sqrt(step_size);
      a[i] = 1./step_size;
      b[i] = -tmpMIN/step_size + degree;
      xvar[i] = xsumsq/nvec - xmean[i]*xmean[i];
    }else{ // no variation in this dimension
      a[i] = -1;
      b[i] = -1;
      st[i] = 0;
      xvar[i] = 0;
    }
  }
  bias = 0; // bias term for the classifier
}

// train the model using LIBLINEAR's dual coordinate descend algorithm
// the learned model is L2 regularized, L1 loss (hinge loss) SVM
void additiveModel::train(double **x,             // training data
			  const double *y,        // training labels
			  const int nvec,         // number of training data
			  const parameter *param) // training parameters
{
	
  //initialize training parameters
  double * alpha = new double[nvec];
  double * Q = new double[nvec];
  int * index = new int[nvec]; 
	
  int i,j,k,s,iter=0,active_size=nvec,wo;
	
  const int MAX_OUTER_ITERS = 1000;
	
  //encoding related variables
  int ei;
  double *ew = NULL, *xi;
  double *xd = new double[numbasis]; //store the dense features
  if(encoding == SPLINE){
    ew = new double[degree+1];
  }	
	
  // initialize the encodings, alpha, index, Q, ...
  for(i = 0; i < nvec;i++){
    alpha[i]=0;
    index[i]=i;
    Q[i] = param->bias * param->bias;
    xi = x[i];
    
    for(j=0; j < dim ; j++){
      if(st[j] > 0){
	if(encoding == SPLINE){
	  bSplineEncoder(xi[j],j,ei,ew);
	  if(reg == 0){ //identity matrix 
	    for(k=0;k<= degree;k++){
	      Q[i] += st[j]*st[j]*ew[k]*ew[k];
	    }
	  }else{ //D_d matrix regularization
	    projectDense(ei,ew,st[j],xd);
	    for(k=0; k < numbasis;k++)
	      Q[i] += xd[k]*xd[k];
	  }
	}else if(encoding == TRIGONOMETRIC){
	  trigEncoder(xi[j],j,xd);
	  for(k=0; k < numbasis;k++)
	    Q[i] += xd[k]*xd[k];
	}else if(encoding == HERMITE){
	  hermiteEncoder(xi[j],j,xd);
	  for(k=0; k < numbasis;k++)
	    Q[i] += xd[k]*xd[k];
	}
      }
    }
  }
  
  double C,d,G;
  // PG: projected gradient, for shrinking and stopping (see LIBLINEAR)
  double PG;
  double PGmax_old = INF;
  double PGmin_old = -INF;
  double PGmax_new, PGmin_new;
  while(iter < MAX_OUTER_ITERS){
    PGmax_new = -INF;
    PGmin_new = INF;
		
    for (i=0; i<active_size; i++){
      int j = i+rand()%(active_size-i);
      swap(index[i], index[j]);
    }
		
    for (s=0;s<active_size;s++){
      i = index[s];
      G = bias*param->bias;
      schar yi = (schar)y[i];
      xi = x[i];
      wo = 0;
      for(j = 0; j < dim ; j++){ //compute the gradient
	if(st[j] > 0){
	  if(encoding == SPLINE){
	    bSplineEncoder(xi[j],j,ei,ew);
	    for(k=0; k <= degree; k++)
	      G += st[j]*w[wo+ei-k]*ew[degree-k]; //sparse (implicit wd)
	  }else if(encoding == TRIGONOMETRIC){
	    trigEncoder(xi[j],j,xd);
	    for(k=0; k < numbasis; k++)
	      G += w[wo+k]*xd[k]; 
	  }else if(encoding == HERMITE){
	    hermiteEncoder(xi[j],j,xd);
	    for(k=0; k < numbasis; k++)
	      G += w[wo+k]*xd[k]; 
	  }
	}
	wo += numbasis;
      }
      G = G*yi-1;
      
      if(yi == 1) 
	C = param->Cp;
      else 
	C = param->Cn;
      
      PG = 0;
      if (alpha[i] == 0){
	if (G > PGmax_old){
	  active_size--;
	  swap(index[s], index[active_size]);
	  s--;
	  continue;
	}
	else if (G < 0)
	  PG = G;
      }
      else if (alpha[i] == C){
	if (G < PGmin_old){
	  active_size--;
	  swap(index[s], index[active_size]);
	  s--;
	  continue;
	}
	else if (G > 0)
	  PG = G;
      }
      else
	PG = G;
      
      PGmax_new = max(PGmax_new, PG);
      PGmin_new = min(PGmin_new, PG);
      
      if(fabs(PG) > 1.0e-12){
	double alpha_old = alpha[i];
	alpha[i] = min(max(alpha[i] - G/Q[i], 0.0), C);
	d = (alpha[i] - alpha_old)*yi;
	wo = 0;
	for(j = 0; j < dim ; j++) {
	  if(st[j] > 0){
	    if(encoding == SPLINE){
	      bSplineEncoder(xi[j],j,ei,ew);
	      if(reg == 0){ //identity matrix 
		for(k=0; k <= degree;k++){
		  w[wo+ei-k] += d*st[j]*ew[degree-k];
		}
	      }
	      else{ //D_d matrix regularization
		projectDenseW(ei,ew,st[j],xd);
		for(k=0; k < numbasis;k++)
		  w[wo+k] += d*xd[k];
	      }
	    }else if(encoding == TRIGONOMETRIC){
	      trigEncoder(xi[j],j,xd);
	      for(k=0; k < numbasis; k++)
		w[wo+k] += d*xd[k]; //sparse (implicit wd)
	    }else if(encoding == HERMITE){
	      hermiteEncoder(xi[j],j,xd);
	      for(k=0; k < numbasis; k++)
		w[wo+k] += d*xd[k]; //sparse (implicit wd)
	    }
	  }
	  wo += numbasis;
	}
	bias += d*param->bias;
      }
    }
    iter++;
    if(iter % 10 == 0)
      printf(".");
    
    if(PGmax_new - PGmin_new <= param->eps){
      if(active_size == nvec)
	break;
      else{
	active_size = nvec;
	printf("*");
	PGmax_old = INF;
	PGmin_old = -INF;
	continue;
      }
    }
    PGmax_old = PGmax_new;
    PGmin_old = PGmin_new;
    if (PGmax_old <= 0)
      PGmax_old = INF;
    if (PGmin_old >= 0)
      PGmin_old = -INF;
  }//outer iteration
  printf("done.\n");
	
  //fold the bias into the model
  bias = bias*param->bias;  	
	
  delete [] alpha;
  delete [] index;
  delete [] Q;
  delete [] xd;
  if(ew != NULL)
    delete [] ew;
}//end of pwltrain

// piecewise linear predictions using the trained model
void additiveModel::predict(double **x,
			    double *d,
			    double *l,
			    const int nvec){
  int i, j, k, wo, ei;
  double di;
	
  //weights for encoding
  double *ew=NULL, *xd = NULL;
  if(encoding == SPLINE)
    ew = new double[degree+1];
  else {
    xd = new double[numbasis];
  }

	
  for(i = 0; i < nvec; i++){
    wo=0;
    di=bias;
		
    for(j = 0; j < dim ; j++){
      if(st[j] > 0){
	if(encoding == SPLINE){
	  bSplineEncoder(x[i][j],j,ei,ew);
	  for(k=0; k <= degree; k++)
	    di += st[j]*w[wo+ei-k]*ew[degree-k]; 
	}else if(encoding == TRIGONOMETRIC){
	  trigEncoder(x[i][j],j,xd);
	  for(k=0; k < numbasis; k++)
	    di += w[wo+k]*xd[k]; 
	}else if(encoding == HERMITE){
	  hermiteEncoder(x[i][j],j,xd);
	  for(k=0; k < numbasis; k++)
	    di += w[wo+k]*xd[k]; 
	}
      }
      wo += numbasis;
    }
    d[i] = di;
    l[i] = di >= 0? 1.0 : -1.0;
  }
  if(ew != NULL)
    delete [] ew;
  if(xd != NULL)
    delete [] xd;
}
//compute the accuracy
void additiveModel::getAccuracy(double *d, double *y, const int nvec,
				int& numcorrect, 
				double&acc, 
				double& prec, 
				double& recall){
  int tp=0, fp=0, tn=0, fn=0, numpos=0;
  for(int i = 0; i < nvec; i++){
    if(y[i] >= 0){ 
      numpos++;
      d[i] >=0?tp++:fn++;
    }else{
      d[i] < 0?tn++:fp++;
    }
  }
  numcorrect = tp+tn;
  acc = (tp+tn)*100.0/nvec;
  prec = tp*100.0/(tp+fp);
  recall = tp*100.0/numpos;
}


// compute the projection of features on the implicit weight vector
// xd = D_d^{-1}D_d^{'-1}\Phi(x)
void additiveModel::projectDenseW(int ei,
				  double *x,
				  double st,
				  double *xd){
  projectDense(ei,x,st,xd);
  for(int d=1;d<=reg;d++){
    for(int k = 1; k < numbasis ;k++)
      xd[k] = xd[k] + xd[k-1];
  }
}
// compute the dense features corresponding to the regularization
// xd = D_d^{'-1}\Phi(x)
void additiveModel::projectDense(int ei,
				 double *x,
				 double st,
				 double *xd){
  //initialize 
  for(int i=0; i < numbasis; i++)
    xd[i] = 0;
	
  for(int i=0; i <= degree; i++)
    xd[ei-i] = st*x[degree-i];
	
  //repeat for various regularization degrees
  for(int d=1; d <= reg ; d++){
    for(int i = numbasis-2;i>=0;i--)
      xd[i] = xd[i] + xd[i+1];
  }
}

// display the model
void additiveModel::display()
{
  printf("Printing model:\n");
  printf("       encoding = %d\n", encoding);
  printf("   basis degree = %d\n", degree);
  printf(" reguralization = %d\n", reg);
  printf("        numbins = %d\n", numbins);
  printf("       numbasis = %d\n", numbasis);
  printf("       data dim = %d\n", dim);
  printf("     weight dim = %d\n", wdim);
	
  printf("            min = ");
  for(int i = 0; i<dim;i++)
    printf("%.4f ", xmin[i]);
  printf("\n");
	
  printf("            max = ");
  for(int i = 0; i<dim;i++)
    printf("%.4f ", xmax[i]);
  printf("\n");
	
	
  printf("           mean = ");
  for(int i = 0; i<dim;i++)
    printf("%.4f ", xmean[i]);
  printf("\n");
	
  printf("            var = ");
  for(int i = 0; i<dim;i++)
    printf("%.4f ", xvar[i]);
  printf("\n");
	
  printf("             st = ");
  for(int i = 0; i<dim;i++)
    printf("%.4f ", st[i]);
  printf("\n");
	
  printf("              a = ");
  for(int i = 0; i<dim;i++)
    printf("%.4f ", a[i]);
  printf("\n");
	
  printf("              b = ");
  for(int i = 0; i<dim;i++)
    printf("%.4f ", b[i]);
  printf("\n");
	
  int wo = 0; 

  for(int i = 0; i<dim;i++){
    printf("      wts [dim=%d] ",i);
    for(int j = 0;j < numbasis;j++){
      printf("%.4f ",w[wo++]);
    }
    printf("\n");
  }
  printf("           bias = %.2f \n", bias);
	
}

// destructor
additiveModel::~additiveModel()
{
  delete [] xmax;
  delete [] xmin;
  delete [] xmean;
  delete [] xvar;
  delete [] st;
  delete [] a;
  delete [] b;
  delete [] w;
  if(dimwts != NULL)
    delete [] dimwts;
}



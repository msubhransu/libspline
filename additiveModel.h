#ifndef _ADDITIVEMODEL_H
#define _ADDITIVEMODEL_H


// problem parameters
struct parameter {
	//encoding type(0 splines, 1 Fourier, 2 Hermite) 
	int encoding; 
	
	// basis degree (0 constant, 1 linear, 2 quadratic, 3 cubic)
	int degree;
	
	// regularization degree (0 identity, 1 first derviative, 2 second derivative)
	int reg; 
	
	//number of pices of spline (#knots-1)
	int numbins;
	
	//bias term added to the data
	double bias;
	
	//regularization parameters for pos/neg
	double Cp;
	double Cn;
	
	//dual violation epsilon
	double eps; 
	
};

// possible encodings
enum ENCODING {SPLINE, TRIGONOMETRIC, HERMITE}; 

//spline model
class additiveModel{
	
public:
	//encoding type(0 splines, 1 Fourier, 2 Hermite) 
	int encoding; 
	
	// basis degree (0 constant, 1 linear, 2 quadratic, 3 cubic for splines|deg for Fourier,Hermite )
	int degree;
	
	// regularization degree (0 identity, 1 first derviative, 2 second derivative)
	int reg; 
	
	//number of divisions of the data
	int numbins;
	
	//number of basis functions
	int numbasis;
	
	// training data dimensions
	int dim; 
	
	// weight dimension
	int wdim;
	
	//min and max of each data dimension
	double *xmin;
	double *xmax;
	
	//mean and variance of each data dimesion
	double *xmean;
	double *xvar;
	
	//learned model and bias
	double *w; 
	double bias;
	
	//parameters for the encoding
	double *st; //square root step size
	double *a; 
	double *b;

	// weights to normalize the dimensions
	double *dimwts;
	
	additiveModel();
	additiveModel(const parameter *, double **, int, int);
	void train(double **x, const double *, const int, const parameter *);
	void predict(double **, double *, double *, const int);
	void getAccuracy(double *, double *, const int, int&, double&, double&, double& );
	void display();
	void getBasisIndex(double x, int dimidx, int &ei, double &ai);
	void bSplineEncoder(double x, int dimidx, int &ei, double *wts);
	void trigEncoder(double x, int dimidx, double *wts);
	void hermiteEncoder(double x, int dimidx, double *wts);
	void projectDense(int ei, double *wts, double st, double *wd);
	void projectDenseW(int ei, double *wts, double st, double *wd);

	~additiveModel();
};
#endif


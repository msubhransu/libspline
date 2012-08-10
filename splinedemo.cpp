#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "splinemodel.h"

void print_usage(){
	printf("usage: ./splinedemo [-r radius] [-b basis degree] [-d regularizaiton] [-n bins]\n"
		   "Generates points on a uniform grid and learns the function x^2 + y^2 < 1.\n"
		   "options:\n"
		   "\t-r radius   : number of points (in each dimension 2r+1)\n"
		   "\t-e encoding : 0 Spline, 1 Fourier, 2 Hermite (see README)\n"
		   "\t-b numbins  : number of spline pieces\n"
		   "\t-d          : display training data\n");
}

int main(int argc, char ** argv){
	int i=1, radius=15, degree = 1, reg=1, numbins = 5, encoding = SPLINE;
	bool display = false;
	if(argc > 1){
		while(i < argc){
			if(strcmp(argv[i],"-r")==0){
				radius = atoi(argv[i+1]);
				if(radius < 0){
					printf("Error: invalid radius (r > 1)\n"); 
					return -1;
				}
			}
			if(strcmp(argv[i],"-b")==0){
				degree = atoi(argv[i+1]);
			}
			if(strcmp(argv[i],"-d")==0){
				reg = atoi(argv[i+1]);
				if(reg < 0){
					printf("Error: invalid regularization (-d 0,1,2,3,...)\n"); 
					return -1;
				}
					
			}
			if(strcmp(argv[i],"-n")==0){
				numbins = atoi(argv[i+1]);
				if(numbins < 1){
					printf("Error: invalid number of bins (>=1)\n"); 
					return -1;
				}
			}
			if(strcmp(argv[i],"-e")==0){
				encoding = atoi(argv[i+1]);
				if(encoding < 0 || encoding > 2){
					printf("Error: invalid encoding (-e 0,1,2)\n"); 
					return -1;
				}
			}
			if(strcmp(argv[i],"-v")==0)
				display = true;
				
			i++;
		}
	}else {
		print_usage();
		return -1;
	}
	

	int n = 2*radius + 1; 
	double **x = new double*[n*n];
	
	double px,py,r;
	for(i = 0; i < n*n; i++){
		x[i] = new double[2];
		px = floor(i/n) - radius; py = i%n-radius;
		x[i][0] = px/radius; x[i][1] = py/radius;
	}

	double *y = new double[n*n];
	for(i = 0; i < n*n ; i++){
		r = x[i][0]*x[i][0] + x[i][1]*x[i][1];
		if(r < 1)
			y[i] =  1;
		else
			y[i] = -1;
	}
	
	parameter *param = (parameter *)malloc(sizeof(parameter));
	//create default values
	param->encoding = encoding;
	param->degree = degree;
	param->reg = reg;
	param->numbins = numbins;
	param->bias = 10;
	param->Cp =  10;
	param->Cn =  10;
	param->eps = 0.1;

	splineModel *model = new splineModel(param,x,2,n*n);
	
	time_t start_time, end_time;
	
	// start training
	start_time = clock();
	model->splineTrain(x,y,n*n,param);
	end_time = clock();
	printf("%.2fs for training on %i points\n", (end_time - start_time)*1.0/CLOCKS_PER_SEC,n*n);
	//show model
	model->display();
	
	//compute predictions
	double *d = new double[n*n];
	double *l = new double[n*n];
	
	start_time = clock();
	model->splinePredict(x,d,l,n*n);
	end_time = clock();
	printf("%.2fs for prediction on %i points\n", (end_time - start_time)*1.0/CLOCKS_PER_SEC,n*n);
	
	if(display){
		printf("Training data (+ true pos, - true neg, o misclassified):\n");
		int idx = 0;
		for(int i = 0; i < n ; i++){
			for(int j = 0; j < n; j++){
				if(y[idx] > 0){
					if(l[idx] > 0)
						printf("+");
					else 
						printf("o");
				}
				else{
					if(l[idx] <= 0)
						printf("-");
					else 
						printf("o");
				}
				idx++;
			}
			printf("\n");
		}
	}
	//compute accuracy on training data
	int nc; double acc, prec, recall;
	model->getAccuracy(d,y,n*n, nc,acc,prec,recall);
	printf("Training accuracy: %.2f%% (%i/%i), %.2f%%/%.2f%% (precision/recall)\n",acc, nc, n*n,prec,recall); 
		   //clean up allocated data
	delete [] y;
	delete [] d;
	delete [] l;
	
	for(i = 0; i < n*n; i++)
		delete [] x[i];
	
	delete [] x;
	
	free(param);
	delete model;
}

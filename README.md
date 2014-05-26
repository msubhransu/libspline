## LIBSPLINE: Max-Margin Additive Classifiers

Created by Subhransu Maji

LIBSPLINE is library for training max-margin additive classifiers based on splines and other representations. The solver is based on LIBLINEAR's dual coordinate 
descend algorithm which scales well to large-scale classification problems. The embeddings are computed online to minimize memory overhead during training and evaluation. For details please refer to the papers at the end of the README, in particular the ECCV'12 workshop paper.

### Compiling

Assuming you have mex compiler set up in your MATLAB run `compile.m` at the prompt.

### Usage

The library is interfaced with MATLAB and provides three functions:

* `train` 	: trains additive classifiers
* `predict` : predicts labels using trained models
* `encode` 	: encode features `x -> Φ(x)`, such that the decision function is `w'Φ(x)`. These can be used directly with linear solvers such as [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/).

Below is the syntax for each of the commands.

#### Train
<pre>
model = train(training_label_vector, training_instance_matrix, 'options', 'col');
options:
-t type     : 0: Spline, 1: Trigonometric, 2: Hermite  (default=0)
-d degree   : set the B-Spline degree (default=1) d={0,1,2,3}
-r reg      : set the order of regularization (default=1) r={0,1,2,...}
-n bins     : set the number of bins (default 10)
-c cost     : set the parameter C (default 1)
-e epsilon  : set tolerance of termination criterion
		       Dual maximal violation <= eps; similar to libsvm (default 0.1)
-B bias     : if bias >= 0, instance x becomes [x; bias]; if bias < 0, no bias term is added (default 1)
-wi weight  : weights adjust the parameter C of different classes (see README for details)
col:
	if 'col' is set, training_instance_matrix is parsed in column format, otherwise is in row format
</pre>	
#### Predict	

<pre>
[predicted_label,accuracy,decision_values] = predict(test_label_vector, test_instance_matrix, model, 'col');
options:
	if 'col' is set, test_instance_matrix is parsed in column format, otherwise is in row format
</pre>

#### Encode
<pre>
[encodedFeats, model] = encode(feats, [model or 'options'], 'col');
outputs:
encodedFeats	: encoded features (Note, these are in 'col' format) 
model		: if options are provided instead of a model, then returns a model

options:
-t type		: O: Spline, 1: Trigonometric, 2: Hermite (default=0, t={0,1,2} )
-d degree	: set the degree of B-Spline basis (default=1, d={0,1,2,3} )
-r reg		: set the order of regularization (default=1) r={0,1,2,...}
-n bins		: set the number of bins (default 10)
</pre>


### References
If you find the code useful please consider citing:

	@inproceedings{maji09iccv, 
				author = {Subhransu Maji and Alexander Berg}, 
				title = {Max-margin additive classifiers for detection}, 
				booktitle = {International Conference on Computer Vision, ICCV}
				year = 2009}
				
	@inproceedings{maji2012eccv, 
				author = {Subhransu Maji}, 
				title = {Linearized Smooth Additive Classifiers},
				booktitle = {Workshop on Web-scale Vision and Social Media, ECCV}, 
				year = 2012}
				
	@inproceedings{maji2013pami, 
				author={Subhransu Maji and Alexander Berg and Jitendra Malik},
				title = {Efficient Classification for Additive Kernel SVMs},
				booktitle = {IEEE Transactions of PAMI},
				year = 2013,
				month = January}

<i>For any questions and comments email `smaji@cs.umass.edu`</i>

% Simple demo file for using LIBSPLINE library. 
% 
% Learns the decision function x^2 + y^2 < 1 for points uniformly
% distributed on a unit square.
%
% Three models are trained (1) Cubic Spline with D_1 regularization (2)
% Trigonometric embedding with D_1 regularization (3) Hermite embedding
% with D_1 regularization
%
% Author: Subhransu Maji (smaji@ttic.edu)


% Generate training data %
r = 20;
fprintf('Learning the function x^2 + y^2 < 1\n');
[XX,YY] = meshgrid(-1:1/r:1,-1:1/r:1);
X = [XX(:) YY(:)];
Y = 2*(sqrt(X(:,1).^2 + X(:,2).^2) < 1)-1 ;



figure;hold on;
set(gca,'FontSize',16);    
for i = 1:length(Y)
    if(Y(i) > 0)
        plot(X(i,1),X(i,2),'g.');
    else
        plot(X(i,1),X(i,2),'r.');
    end
end

% Draw the decision boundary
theta = linspace(0,2*pi,100);
plot(cos(theta),sin(theta),'k-');
xlabel('x');
ylabel('y')
title(sprintf('Training data (%i points)',length(Y))); 
axis equal tight; box on;
clc;

% Learn various additive models and draw the decision boundary%
encoding = [0 1 2];
linecolor = 'rgbk';
figure; hold on;
set(gca,'FontSize',16);    
fprintf('\n\nTraining various additive models (%i points)\n', length(Y));
for j = 1:length(encoding),
    tic;
    paramstr = sprintf('-t %i -d 3 -r 2 -n 10 -B 1 -c 10',encoding(j));
    model = train(Y,X,paramstr);
    [l,a,d] = predict(Y,X,model);
    plot(d(1:2*r+1),[linecolor(j) '-'],'LineWidth',2);
    fprintf('Training : %s , %.2fs elapsed, Accuracy : %.2f%%\n', paramstr,toc,a(1));
end
xlabel('x');
ylabel('f(x,y)');
title('Learned f(x,y) using various embeddings');
axis tight; box on;
legend('Cubic B-Spline','Fourier', 'Hermite', 'Location', 'South');

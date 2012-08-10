fprintf('Learn the function x^4 + y^4 < 1\n');

r = 20;

if(r <= 20) 
    DISPLAY = 1; 
else
    DISPLAY = 0;
end
   
[XX,YY] = meshgrid(-1:1/r:1,-1:1/r:1);
X = [XX(:) YY(:)];
Y = 2*(sqrt(X(:,1).^4 + X(:,2).^4) < 1)-1 ;
if DISPLAY
    figure;hold on;
    for i = 1:length(Y)
        if(Y(i) > 0)
            plot(X(i,1),X(i,2),'g.');
        else
            plot(X(i,1),X(i,2),'r.');
        end
    end
    %draw the decision boundary
    theta = linspace(0,2*pi,100);
    plot(cos(theta),sin(theta),'k-');
    title(sprintf('Training data (%i points)',length(Y))); axis equal tight; box on;
end
%% start training
encoding = [0 1 2];
linecolor = 'rgbk';
figure; hold on;
    

fprintf('\n\nBenchmarking various encoding schemes (%i points)\n', length(Y));
for j = 1:length(encoding),
    tic;
    paramstr = sprintf('-t %i -d 3 -r 1 -n 10 -B 1 -c 10',encoding(j));
    model = splinetrain(Y,X,paramstr);
    [l,a,d]=splinepredict(Y,X,model);
    plot(d(1:2*r+1),[linecolor(j) '-'],'LineWidth',2);
    fprintf('Training : %s , %.2fs elapsed, Accuracy : %.2f%%\n', paramstr,toc,a(1));
end
axis tight; box on;
legend('Cubic B Spline','Fourier', 'Hermite');
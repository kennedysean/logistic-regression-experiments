% import X_train, X_test, y_train, y_test from ad_data.mat
load('ad_data.mat')
% run sparse logistic regression with each regularization parameter in pars
pars  = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
slr_results = zeros(size(pars,2),4);
for i=1:size(pars,2)
    % train sparse logistic regression 
    [w,c] = logistic_l1_train(X_train,y_train,pars(i));
    correct_count = 0; % number of correctly classified data points
    score = zeros(size(y_test,1),1);
    for j=1:size(y_test,1) % test predictions 
        class = [X_test(j,:) ones(1,1)] * [w;c] ;
        score(j) = class;
        if (class > 0 && y_test(j) == 1) || (class < 0 && y_test(j) == -1)
            correct_count = correct_count + 1;
        end
    end
    % get sparse logistic regression results
    [X,Y,T,AUC] = perfcurve(y_test,score, 1);
    slr_results(i,:) = [pars(i) nnz(w) AUC 100*correct_count/size(y_test,1)];
end
disp('Regularization Parameter | Number of Features Selected | AUC | Accuracy (%)')
disp(slr_results)
writematrix(slr_results, 'slr_results.csv')
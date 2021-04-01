training = importdata('training.txt');
test = importdata('test.txt');
n_arr = [200, 500, 800, 1000, 1500, 2000];
lr_results = zeros(size(n_arr,2),3);
% training on n data points for values in n_arr
for n_index=1:size(n_arr,2)
    n = n_arr(n_index);
    intercepts = ones(n,1);
    data = [training(1:n,1:57) intercepts];
    labels = training(1:n,58);
    epsilon = 0.00001;
    maxiter = 1000;
    w = logistic_train(data,labels, epsilon, maxiter);
    correct_count = 0; % number of correctly classified emails
    for i=1:size(test,1)
        class = [test(i,1:57) ones(1,1)] * w;
        if class < 0 & test(i,58) == 0
            correct_count = correct_count + 1;
        end
        if class > 0 & test(i,58) == 1
            correct_count = correct_count + 1;
        end
    end
    lr_results(n_index,:) = [n, correct_count correct_count/size(test,1)];
end
format shortG
disp('      N | Number correctly classified | Classification Accuarcy')
disp(lr_results)
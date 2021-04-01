%
% Code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
function [weights] = logistic_train(data, labels, epsilon, maxiter)
    % default parameters
    if nargin == 2
        epsilon = 0.00001;
        maxiter = 1000;
    end
    if nargin == 3
        maxiter = 1000;
    end
    % Initialize weights
    weights = zeros(size(data,2),1);
    % Perform gradient descent
    while maxiter > 0 
        old_weights = weights;
        weights = old_weights - gradient(old_weights, data, labels);
        maxiter = maxiter - 1;
        if mean(abs(weights-old_weights)) < epsilon
            break
        end
    end
end

function [grad] = gradient(w, data, labels)
    % Calculate gradient at w
    N = size(data,1); 
    M = size(data,2);
    sum = zeros(M,1);
    for i=1:N
        x=data(i,:);
        sig = 1/(1+exp(-x*w));
        sum = sum + ((labels(i,1) - sig)*transpose(x));
    end
    grad = (-1/N)*sum;
end


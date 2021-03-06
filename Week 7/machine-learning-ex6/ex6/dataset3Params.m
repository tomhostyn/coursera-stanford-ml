function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

do_optimisation = 0

if (do_optimisation)

  %C_candidates = [1, 10];
  C_candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
  %sigma_candidates = [0.01, 0.03];
  sigma_candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

  %generate all possible pairs
  [p,q] = meshgrid(C_candidates, sigma_candidates);

  combinations = size(p(:))(1);

  %all combinations + a slot for the prediction error
  pairs = [p(:) q(:) zeros(combinations,1)];

  pairs

  for i = 1:combinations
    C = pairs(i, 1);
    sigma = pairs (i, 2);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    pairs(i, 3) = mean(double(predictions ~= yval));
  end

  pairs
    
  [m i ] = min (pairs(:, 3));
  C = pairs(i, 1)
  sigma = pairs (i, 2)

end  
% =========================================================================

end

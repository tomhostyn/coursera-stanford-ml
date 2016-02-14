function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% calculate cost by Forward propagation 

% calculate the hypothesis h 
% estimate probablity for every class given every sample
% DIM: m (number of samples) x num_labels (number of classes)

%h1 = sigmoid([ones(m, 1) X] * Theta1');
%h = sigmoid([ones(m, 1) h1] * Theta2');

%breakup in steps useful for backpropagation later
A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);
Z3 = [ones(m, 1) A2] * Theta2';
A3 = sigmoid(Z3);
H = A3;

% prep the truth labels in the right format
% every element in y becomes a column with all 0 and a single 1 denoting the right output class
% DIM: Y = m (nr of samples) x num_labels
Y=[];
for i = y
  Y = [Y eye(num_labels)(:, i)];
end
Y = Y';

% cost without regularization
J = 1/m * sum(sum(-Y .* log(H) - (1-Y) .* log(1-H)));

% regularization term.  remove bias terms.

R = lambda/(2 * m) * ...
    (sum (sum(Theta1(:,2:columns(Theta1)) .^2)) + ...
     sum (sum(Theta2(:,2:columns(Theta2)) .^2))); 

J = J + R;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


%breakup in steps useful for backpropagation later

%z2 = [ones(m, 1) X] * Theta1';
%a2 = sigmoid(z2);
%z3 = [ones(m, 1) a2] * Theta2';
%a3 = sigmoid(z3);

% iterate over all samples
t=1;
for t = 1:m
% pick up only current sample t

a3 = A3(t, :);
a2 = [1 A2(t, :)];
a1 = A1(t, :);

y  = Y(t, :);
z2 = Z2(t, :);

% add the bias term to z to fix dimensions.  will be ignored later
z2 = [1 z2];

% delta for output nodes.  dim: num_labels x 1
d3 = (a3 - y)';

d2 = (Theta2' * d3)' .* sigmoidGradient(z2);
d2 = d2(2:end); % drop bias term

Theta1_grad = Theta1_grad + (a1'*d2)' ;
Theta2_grad = Theta2_grad + (a2'*d3')';
end

Theta1_grad = 1/m * (Theta1_grad);
Theta2_grad = 1/m * (Theta2_grad);



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% set bias terms to zeros
Theta1_nobias = [zeros(hidden_layer_size,1) Theta1(:, 2:end)];
Theta1_nobias = [zeros(rows(Theta1),1) Theta1(:, 2:end)];
Theta2_nobias = [zeros(rows(Theta2),1) Theta2(:, 2:end)];

Theta1_grad = Theta1_grad +(lambda/m) * Theta1_nobias;
Theta2_grad = Theta2_grad +(lambda/m) * Theta2_nobias;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

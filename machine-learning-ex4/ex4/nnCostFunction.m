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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%size(X)  %  5000x1
%size(y)  % 5000x1
%size(Theta1)   % 25x401
%size(Theta2)   % 10x26

% transform y to a matix 5000x10
y_matrix = zeros(m,num_labels);
y_matrix(sub2ind([m,num_labels],[1:m]',y)) = 1;   %5000x10

% propagation to the hidden layer
X = [ones(m, 1) X];  % 5000x401

z2 = X*Theta1';        %5000x401 * 401x25  = 5000x25
a2 = sigmoid(z2);     % 5000x25
a2 = [ones(m,1) a2];   % 5000x26

% to the output layer
z3 = a2*Theta2';      % 5000x26 * 26x10 = 5000*10
a3 = sigmoid(z3);    % 5000*10

h = a3;   % 5000x10

%for k = 1:num_labels,
  J = 1/m*sum(sum(-y_matrix.*log(h) - (1-y_matrix).*log(1-h),2),1);  % 5000x1 .* 5000x1
				%grad = 1/m*X'*(h-y);
%end

  %regularlization
  J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2,2),1) + sum(sum(Theta2(:,2:end).^2,2),1));

  d3 = a3 - y_matrix;   % 5000x10
  %Theta1_grad = a2 .* (1-a2);    % 401x25*
  d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);  %theta2 10x26
% 5000x10 * 10x25 .* 5000x25 = 5000x25
  Delta1 = d2'*X;    % 25x5000 * 5000x401 = 25x401
  Delta2 = d3'*a2;   % 10x5000 * 5000x26 = 10x26

  Theta1_grad = 1/m*Delta1;
  Theta2_grad = 1/m*Delta2;
  
  %regularlization
  Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
  Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

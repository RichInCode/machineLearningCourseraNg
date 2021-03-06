function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%size(grad)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% m is the number of training examples
% n is the number of features (not including bias term)

% theta: n+1 x 1
% X:     m   x n+1
% y:     m   x 1

h = X*theta;   % m x n+1 * n+1 x 1 = m x 1

J = 1/(2*m)*(h-y)'*(h-y);

J = J + lambda/(2*m)*theta(2:end,:)'*theta(2:end,:);

grad = 1/m*X'*(h-y);

%printf("size of grad \n");
%size(grad)
%printf("size of theta \n");
%size(theta)

grad(2:end) = grad(2:end) + lambda/m*theta(2:end);


% =========================================================================

grad = grad(:);

%printf("size of grad \n");
%size(grad)

end

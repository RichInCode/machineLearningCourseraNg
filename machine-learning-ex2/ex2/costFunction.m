function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%size(theta)  %3x1
%size(X)      %100x3
%size(y)      %100x1

h = sigmoid(X*theta);

%size(h)       %100x1


for i = 1:m,
  J = J + 1/m*(-y(i)*log(h(i)) - (1-y(i))*log(1-h(i)));

  %grad = grad + 1/m*X'*(h(i)-y(i));  % 3x1 + (3x100 * (100x1 - 100x1))
end

grad = grad + 1/m*X'*(h-y);  % 3x1 + (3x100 * (100x1 - 100x1))






% =============================================================

end

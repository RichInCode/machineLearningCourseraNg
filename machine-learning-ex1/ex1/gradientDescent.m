function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


%printf("size of theta \n")
%size(theta)  %column vector 2x1
%printf("size of X \n")
%size(X)   % 97x2
%printf("size of y \n")
%size(y)   % 97x1

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


h = X*theta;   % 97x1

temp = X'*(h - y);  % 2x97 * 97x1 = 2x1
%xsize(temp)
	
	  
	  theta = theta - (alpha/m)*temp;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

%J_history(iter)


end

end

function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%  theta1    a
% 25x401    5000x401

% to the hidden layer
zi = X*Theta1';
ai = sigmoid(zi);
ai = [ones(m, 1) ai];  %bias node

% to the output layer
%   theta2    a
%    10x26    5000x26
zf = ai*Theta2';
af = sigmoid(zf);


[dump p] = max(af, [], 2);   % 5000x10




% =========================================================================


end

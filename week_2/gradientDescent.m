function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
p=0;
q=0;
for j=1:m,
p=p+(theta(1)+theta(2)*X(j,2)-y(j));
q=q+(theta(1)+theta(2)*X(j,2)-y(j))*X(j,2);
end
theta1=theta(1)-alpha*(1/m)*p;
theta2=theta(2)-alpha*(1/m)*q;
theta=[theta1;theta2];
%================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

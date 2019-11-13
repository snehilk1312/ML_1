function J = computeCost(X, y, theta)
m = length(y); % number of training examples
J = 0;
p=0;
for i=1:m,
p=p+(theta(1)+theta(2)*X(i,2)-y(i))^2;
J=(1/(2*m))*p;

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
prediction=X*theta;
p=sigmoid(prediction);
a=0;
for i=1:m,
if y(i)==1
a=a-log(p(i));
else
a=a-log(1-(p(i)));
end;
f=a/m;
b=0;
Theta=theta
Theta(1)=0;
b=sum(Theta.^2);
h=b*lambda/(2*m);
J=f+h;
error=p-y;
new=error'*X;
grad=(1/m)*new'+(lambda/m)*theta;
grad2=(1/m)*new';
grad(1)=grad2(1);


% =============================================================

end

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
%
% Note: grad should have the same dimensions as theta
%
z=zeros(1,m);
z=theta' * X' ;
g = sigmoid(z);
for i=1:m,
	J=J-y(i,1)*log(g(1,i))-(1-y(i,1))*log(1-g(1,i));
	z(1,i)=g(1,i)-y(i,1);
end;
J=J/m;
for i=2:size(theta,1),
	J=J+(theta(i,1)^2)*lambda/(2*m);
end;
	
grad=X' * z' ;
grad=grad/m;
for i=2:size(theta,1),
	grad(i,1)=grad(i,1)+lambda*theta(i,1)/m;
end;
% =============================================================

end

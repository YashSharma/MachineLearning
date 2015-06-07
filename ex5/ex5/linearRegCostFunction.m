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

suma=X*theta-y;
suma=suma.^2;
suma=sum(suma)/(2*m);

theta_used=theta(2:end);
sumb=theta_used'*theta_used;
sumb=(sumb*lambda)/(2*m);

J=suma+sumb;

grad(1)=((X*theta-y)'*(X(:,1)))/(m);
grad(2:end)=(((X*theta-y)'*(X(:,2:end))))'/(m)+((theta_used)*(lambda)/(m));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end

function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%X(:,2) = (X(:,2) - (sum(X(:,2)) / m)) / (max(X(:,2)) - min(X(:,2))); %这里是X的特征缩放
%y = (y - sum(y)/m)/(max(y) - min(y)); %这里是y的特征缩放

predictions = X * theta;
sqrErrors = (predictions - y) .^ 2;
J = (1 / (2 * m)) * sum(sqrErrors); %最后的公式计算
% =========================================================================

end

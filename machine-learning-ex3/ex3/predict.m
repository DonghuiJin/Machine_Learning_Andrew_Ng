function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); %求出有多少个手写数字
num_labels = size(Theta2, 1); %求出有多少种数字类别
X=[ones(m,1) X]; %为a1添加为1的偏置
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
for i=1:m, %分别对m个样本做预测
  a2=sigmoid(Theta1*X(i,:)'); %计算a2，size(X(i,:)')=(401*1)，X经过了预处理，多加一列1
  a2=[1;a2];                  %为a2添加为1的偏置
  a3=sigmoid(Theta2*a2);      %计算a3
  [manum index]=max(a3);      %求出哪个数字的预测值最大
  p(i)=index;                 %得出预测值
end
% =========================================================================

end

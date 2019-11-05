function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); %����ж��ٸ���д����
num_labels = size(Theta2, 1); %����ж������������
X=[ones(m,1) X]; %Ϊa1���Ϊ1��ƫ��
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
for i=1:m, %�ֱ��m��������Ԥ��
  a2=sigmoid(Theta1*X(i,:)'); %����a2��size(X(i,:)')=(401*1)��X������Ԥ�������һ��1
  a2=[1;a2];                  %Ϊa2���Ϊ1��ƫ��
  a3=sigmoid(Theta2*a2);      %����a3
  [manum index]=max(a3);      %����ĸ����ֵ�Ԥ��ֵ���
  p(i)=index;                 %�ó�Ԥ��ֵ
end
% =========================================================================

end

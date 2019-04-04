function [trainSet,validationSet,testSet,trainSet_d,validationSet_d,testSet_d] = divideset(X,Y)
% DIVIDESET用于划分训练集、验证集和测试集
total_N = size(X,2);     %求输入的总样本数
trainSet_N = round(0.7*total_N);   %训练集样本数
validationSet_N = round(0.15*total_N);   %验证集样本数
id = randperm(total_N);
trainSet_id = id(:,1:trainSet_N);       %取前trainSet_N列为训练集的样本编号
validationSet_id = id(:,(trainSet_N+1):(trainSet_N+validationSet_N));    %取接着的validationSet_N列为验证集的样本编号
testSet_id = id(:,(trainSet_N+validationSet_N+1):total_N);    %取接着的validationSet_N列为测试集的样本编号
trainSet = X(:,trainSet_id);                   %取出训练集
validationSet = X(:,validationSet_id);         %取出验证集
testSet = X(:,testSet_id);                     %取出测试集
trainSet_d = Y(:,trainSet_id);
validationSet_d = Y(:,validationSet_id);
testSet_d = Y(:,testSet_id);
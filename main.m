%% 清空环境变量
clc
clear

%% 导入数据及定义参数

load('data.mat');
global w1 w2 lr epochs goal max_fail sigma validation;
m =25;    %隐藏层单元数
w1 = rand(m,784)*0.2-0.1;           %输入层到隐藏层的权值,范围在(-0.1,0.1)
w2 = rand(10,m)*0.2-0.1;            %隐藏层到输出层的权值,范围在(-0.1,0.1)
lr =0.02;    %学习速率
epochs = 300;    %最大迭代次数
goal = 0;     %收敛误差
max_fail = 6;   %最大失败次数

%% 训练网络

%划分数据集
[trainSet,validationSet,testSet,trainSet_d,validationSet_d,testSet_d]=divideset(x,d);

%训练网络
mytrain(trainSet,trainSet_d,validationSet,validationSet_d);

%% 结果输出

yc = w1*testSet;
yc = sigmf(yc,[1 0]);      %计算隐藏单元的输出yc
y = w2*yc;
y = sigmf(y,[1 0]);      %计算输出单元的输出y
figure(1)
plotconfusion(testSet_d,y);          %画出混淆矩阵
figure(2)
plot(sigma,'.');     %画出训练过程的误差变化图
hold on
plot(validation,'r+');               %画出验证集的误差变化图
legend('Training set error','Validation set error');     %加上图例

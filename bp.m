function bp(input,d)
%MYTRAIN用于训练BP神经网络，输入参数input是一个样本的数据,d是对应的期望输出
global w1 w2 lr;        %将权值定义为全局变量
yc_output = w1*input;
yc_output = sigmf(yc_output,[1 0]);      %计算隐藏单元的输出yc
output = w2*yc_output;
output = sigmf(output,[1 0]);      %计算输出单元的输出y
output_err = (d-output).*output.*(1-output);     %计算每个输出单元的误差项
yc_err = (w2.'*output_err).*yc_output.*(1-yc_output);       %计算每个隐藏单元的误差项
w1 = w1+lr*yc_err*(input.');   %更新w1权值
w2 = w2+lr*output_err*(yc_output.');   %更新w2权值
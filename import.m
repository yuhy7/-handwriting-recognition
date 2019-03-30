x=cell(1,5000);   %建立一个空元胞数组，1行5000个元素

for i=1:5000
imgpath=strcat('D:\图片\',num2str(i),'.jpg');
x{i}=imread(imgpath);         %导入图片
end

x=cell2mat(x);       %元胞数组转化为矩阵
x=reshape(x,784,5000);      %转换为样本集

x=im2double(x);          %归一化处理

d=importdata('D:\资料\智能信息处理\期望输出.xlsx');    %导入期望输出

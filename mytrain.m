function mytrain(trainSet,trainSet_d,validationSet,validationSet_d)
%MYTRAIN用于训练BP神经网络
global w1 w2 epochs goal max_fail sigma validation
fail = 0;     %当前失败次数
count = 0;      %当前迭代次数
sigma = zeros(1);    %训练集的误差
validation = zeros(1);    %验证集的误差
trainSet_N = size(trainSet,2);
validationSet_N = size(validationSet,2);

while count<epochs         %进行规定次数的迭代

%对全部样本进行遍历，更新权值
i=1;
while i<=trainSet_N
     input = trainSet(:,i);
     expect_d = trainSet_d(:,i);
     i=i+1;
     bp(input,expect_d);
end

count=count+1;     %迭代次数加1

%计算训练集的当前误差
hidden = w1*trainSet;
hidden = sigmf(hidden,[1 0]);
o = w2*hidden;
o = sigmf(o,[1 0]);
e = trainSet_d-o;
sum = 0;
for num = 1:trainSet_N
   E = e(:,num);
   sum = sum+E.'*E;
end
sigma(count) = sum/2;

%运行过程可视化
sprintf('第%d次迭代，误差为%f',count,sigma(count))

%判断是否已达到目标误差，若是，退出迭代过程
if sigma(count) <= goal 
    sprintf('Goal is reached!')
    break
end

%计算验证集误差
v_hidden = w1*validationSet;
v_hidden = sigmf(v_hidden,[1 0]);
v_o = w2*v_hidden;
v_o = sigmf(v_o,[1 0]);
v_e = validationSet_d-v_o;
v_sum = 0;
for v_num = 1:validationSet_N
    v_E = v_e(:,v_num);
    v_sum = v_sum+v_E.'*v_E;
end
validation(count) = v_sum/2;

%判断验证集误差是否连续上升
if count ~= 1
    if validation(count) > validation(count-1)
        fail = fail+1;
    else
        fail = 0;
    end
end

%备份验证误差上升前的权值
if fail == 0
    best_w1 = w1;
    best_w2 = w2;
end

%判断误差是否连续上升若干次迭代，若是则停止训练
if fail >= max_fail
    w1 = best_w1;
    w2 = best_w2;
    sprintf('Validation stop.')
    break
end

end
clear 
clc 

update = 100;
trainTotal = 2000;
testTotal = 720;
hiv_data = csvread('hiv_data.csv', 1, 0);

train = hiv_data(1:2000, 1:8);
test = hiv_data(2001:2720, 1:8);

train_label = hiv_data(1:2000, 9);
test_label = hiv_data(2001:2720, 9);

Beta = 0.01;
theta0 = 1;
theta1 = 0.5;

CN =  [];
for i=1:trainTotal
    
    k = [];
    for j=1:trainTotal
         k=[k theta0*exp(-0.5*(norm(train(i, :)-train(j, :))^2))+theta1];
    end
    CN = [CN; k];
end

CN = CN + eye(trainTotal)*Beta;
t_n = train_label;
a_n = zeros(trainTotal, 1);
w_n = [];

for i=1:update
    
    sigmoid_n = [];
    
    for j=1:trainTotal
        sigmoid_n = [sigmoid_n; sigmf(a_n(j),[1 0])];
    end
    
    w_n = diag(sigmoid_n.*(1-sigmoid_n));
    a_n = CN*inv(eye(trainTotal)+w_n*CN)*(t_n-sigmoid_n+w_n*a_n);
end

K = [];
for i=1:trainTotal
    k = [];
    for j=1:testTotal
        k = [k theta0*exp(-0.5*(norm(train(i, :)-test(j, :))^2))+theta1];
    end
    K = [K; k];
end

m = K'*(t_n-sigmoid_n);

predction = zeros(testTotal, 1);

% make predction, if data <= 0.5 , let it be 1, otherwise, let it be 0
for i=1:testTotal
    if sigmf(m(i), [1 0]) >= 0.5
        predction(i) = 1;
    else
        predction(i) = 0;
    end
end

accuracy = sum(predction == test_label)/testTotal;



clc
clear

trainTotal = 1200;
testTotal = 400;
gp_data = csvread('gp_data.csv');

Beta = 0.01;
theta0 = 1;
theta1 = 0.5
eta0 = 1;
eta1 = 1;

eta = [eta0, eta1];
theta = [theta0, theta1];

train = gp_data(1:1200, :);
test = gp_data(1201:1600, :);

CN = [];

%train
for i=1:trainTotal
    
    k = [];
    for j=1:trainTotal
        k = [k; theta0*exp(-0.5*eta0*(train(i,1)-train(j,1))^2-0.5*eta1*(train(i,2)-train(j,2))^2)+ theta1];
    end
    
    CN = [CN, k];
end

CN = CN + eye(trainTotal)*Beta;

K = [];
% test
for i=1:trainTotal
    k = [];
    for j=1:testTotal
        k = [k; theta0*exp(-0.5*eta0*(train(i,1)-test(j,1))^2-0.5*eta1*(train(i,2)-test(j,2))^2)+ theta1];
    end
    K = [K, k];
end

m = [];
for i=1:testTotal
    m = [m; K(i, :)*inv(CN)*train(:,3)];
end


Erms = sqrt(2*sum((m-test(:,3)).^2)/testTotal);

figure;
scatter3(test(:,1), test(:,2), test(:, 3), 'ro');
hold on
scatter3(test(:,1), test(:,2), m(:), 'bx');

legend('ground truth', 'test');
xlabel('x1');
ylabel('x2');
zlabel('y');


load('x3.mat');
load('t3.mat');

%Read data
X_Train = x3_v2.train_x;
T_Train = t3_v2.train_y;
X_Test = x3_v2.test_x;
T_Test = t3_v2.test_y;

Train_Temp = [];
Test_Temp = [];
Phi = [];
W = [];

%Generate Phi
for i=0:9
    Phi = horzcat(Phi, X_Train(1:15, :).^i);
    Train_Temp = horzcat(Train_Temp, X_Train.^i);
    Test_Temp = horzcat(Test_Temp, X_Test.^i);
end

I = eye(10);
Error_Train = [];
Error_Test = [];

for Lambda = -20:5

    %W = inv(exp(Lambda)*I + Phi.'*Phi)*Phi.'*T_Train;
    W =(exp(Lambda)*I + Phi.'*Phi)\Phi.'*T_Train;
    Error_Train = [Error_Train; rms((Train_Temp*W - T_Train)+ exp(Lambda)/2*norm(W)^2)];
    Error_Test = [Error_Test; rms((Test_Temp*W - T_Test)+ exp(Lambda)/2*norm(W)^2)];
end

%Plot the result
x = [-20:5]; %x-axis
plot(x, Error_Test);
hold on;
plot(x, Error_Train);


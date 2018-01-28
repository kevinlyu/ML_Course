[N,A,rawData] = xlsread('data.xlsx')

Train_X = []; %Training data, 4 Dimention
Train_Y = []; %Y of training data
Test_X = []; %Testing data, 4 Dimention
Test_Y = []; % Y of testing data
Phi = []; %Use pseudo inverse to get matrix 
W = []; %[w0, w1, w2 ...]
Append_Terms = []; % The matrix has elements x11,x12,..

Temp_Row = [];

%Read Train_Xing data from excel file
for i = 1:400
    Train_X = [Train_X; N(i, 1:4)];
    Train_Y = [Train_Y; N(i,5)];
end
%Read Testing data from excel file
for i = 401:500
   Test_X = [Test_X; N(i, 1:4)];
   Test_Y = [Test_Y; N(i,5)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  M=3, Training Stage  %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Generate the terms: x11, x12, x13, ..., x41, x42, x43, x44
for i=1:400
    
    Temp_Row = [];
    M2 = Train_X(i, 1:4).'.*Train_X(i, 1:4);
    
    Temp_Row = horzcat(Temp_Row, reshape(M2, 1, 16));
    
    %Generate the terms: x111, x112, x113, ..., x444
    
    for j=1:4
        Temp_Row = horzcat(Temp_Row, reshape(M2(j, 1:4).* Train_X(i, 1:4).', 1, 16));
    end
    
    Append_Terms = [Append_Terms; Temp_Row];
end

Append_Ones = ones(400,1); %concate to left hand side of Phi
Phi = horzcat(Append_Ones, Train_X, Append_Terms);

W = pinv(Phi.'*Phi)*Phi.'*Train_Y;
t = Phi*W;

Error_train = Train_Y-t;
E_rms_train = rms(Error_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  M=3, Test Stage  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Append_Terms = [];
%Generate the terms: x11, x12, x13, ..., x41, x42, x43, x44
for i=1:100
    
    Temp_Row = [];
    M2 = Test_X(i, 1:4).'.*Test_X(i, 1:4);
    
    Temp_Row = horzcat(Temp_Row, reshape(M2, 1, 16));
    
    %Generate the terms: x111, x112, x113, ..., x444
    
    for j=1:4
        Temp_Row = horzcat(Temp_Row, reshape(M2(j, 1:4).* Test_X(i, 1:4).', 1, 16));
    end
    
    Append_Terms = [Append_Terms; Temp_Row];
end

Append_Ones = ones(100,1); %concate to left hand side of Phi
Phi = horzcat(Append_Ones, Test_X, Append_Terms);
t = Phi*W;

Error_test = Test_Y-t;
E_rms_test = rms(Error_test);

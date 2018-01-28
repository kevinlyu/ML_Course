[N,A,rawData] = xlsread('data.xlsx')

Train_X = []; %Training data, 4 Dimention
Train_Y = []; %Y of training data
Test_X = []; %Testing data, 4 Dimention
Test_Y = []; % Y of testing data
Phi = []; %Use pseudo inverse to get matrix 
W = []; %[w0, w1, w2 ...]
Append_Terms = []; % The matrix has elements x11,x12,..

Temp_Row = [];

Select = [2 3 4] %Select the rows you want to read
Dim = [1 2 3]

%Read Train_Xing data from excel file
for i = 1:400
    Train_X = [Train_X; N(i, Select)];
    Train_Y = [Train_Y; N(i,5)];
end
%Read Testing data from excel file
for i = 401:500
   Test_X = [Test_X; N(i, Select)];
   Test_Y = [Test_Y; N(i,5)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  M=3, Training Stage  %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Generate the terms: x11, x12, x13, ..., x41, x42, x43, x44
for i=1:400
    
    Temp_Row = [];
    M2 = Train_X(i, Dim).'.*Train_X(i, Dim);
    
    Temp_Row = horzcat(Temp_Row, reshape(M2, 1, 9));
    
    %Generate the terms: x111, x112, x113, ..., x444
    
    for j=1:3
        Temp_Row = horzcat(Temp_Row, reshape(M2(j, Dim).* Train_X(i, Dim).', 1, 9));
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
    M2 = Test_X(i, Dim).'.*Test_X(i, Dim);
    
    Temp_Row = horzcat(Temp_Row, reshape(M2, 1, 9));
    
    %Generate the terms: x111, x112, x113, ..., x444
    
    for j=1:3
        Temp_Row = horzcat(Temp_Row, reshape(M2(j, Dim).* Test_X(i, Dim).', 1, 9));
    end
    
    Append_Terms = [Append_Terms; Temp_Row];
end

Append_Ones = ones(100,1); %concate to left hand side of Phi
Phi = horzcat(Append_Ones, Test_X, Append_Terms);
t = Phi*W;

Error_test = Test_Y-t;
E_rms_test = rms(Error_test);

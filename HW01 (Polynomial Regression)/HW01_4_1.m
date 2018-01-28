load('x3.mat');
load('t3.mat');

%Data for Train Stage
X_Train = x3_v2.train_x;
T_Train = t3_v2.train_y;

%Data for Test Stage
X_Test = x3_v2.test_x;
T_Test = t3_v2.test_y;

%Devide data into 3 groups
Group = [[1:5, 6:10]; [1:5, 11:15]; [6:10, 11:15]];

%Error from M=1 to M=9 of all three sets
Error_Train = [];
Error_Test = [];
Select_Group = [];
Train_final =[];
Test_final = [];

W = [];
Validation_Error = [];

for M = 1:9
    %Reset
    Error_Cross = intmax;

    for g = 1:3
        
        %first, cross validation
        Phi = [];
        
        Train_Temp = [];
        Test_Temp = [];
        Train_Cross = [];
        
        for i = 0:M
            Phi = horzcat(Phi, X_Train(Group(g,:)).^i); %Generate Phi from X_Train
            
            Train_Cross = horzcat(Train_Cross, X_Train(setdiff([1:15], Group(g,:))).^i);
            
            Train_Temp = horzcat(Train_Temp, X_Train(Group(g,:)).^i);
            Test_Temp = horzcat(Test_Temp, X_Test.^i);
        end
 
        Wtmp = (Phi.'*Phi)\Phi.'*T_Train(Group(g,:)); % Model trained
        %left divide
        %Wtmp = (Phi.'*Phi)\Phi.'*T_Train(Group(g,:)); % Model trained
        
        %Cross validate
        Error_g = rms(Train_Cross*Wtmp - T_Train(setdiff([1:15], Group(g,:))));
        Validation_Error = [Validation_Error; Error_g];
        
        if(Error_g < Error_Cross)
            %Get the better model and its W, X_Train, T_Train
            Error_Cross = Error_g;
            W = Wtmp;
            Select_Group = Group(g,:);
            Train_final = Train_Temp;
            Test_final = Test_Temp;
        end
    end
    %Error_Train = [Error_Train; Error_Cross];
    Error_Train = [Error_Train; rms(Train_final*W - T_Train(Select_Group,:))];
    Error_Test = [Error_Test; rms(Test_final*W - T_Test)];
    
end

x = [1:8];
plot(x, Error_Train(1:8));
hold on;
plot(x, Error_Test(1:8));
xlabel('M = 1 to 9');
ylabel('E RMS');
legend('Train','Test');  

clear
clc

load('Iris.mat')

move = 0.5
trainTotal = 120;
testTotal = 30;
C = 1000;                                                                                           
tolerance = 0.001;

label1 = -ones(trainTotal, 1);
label2 = -ones(trainTotal, 1);
label3 = -ones(trainTotal, 1);

% Mark the selected class as 1 
label1(1:40) = 1;
label2(41:80) = 1;
label3(81:120) = 1;

%%%%%%%%%%% train stage %%%%%%%%%%%
linear_kernel = trainFeature * trainFeature';

[alpha1, bias1] = smo(linear_kernel, label1', C, tolerance);
[alpha2, bias2] = smo(linear_kernel, label2', C, tolerance);
[alpha3, bias3] = smo(linear_kernel, label3', C, tolerance);

% record which point is support vector
sv_idx = [];

for i=1:120
    if alpha1(i) ~= 0
       sv_idx = [sv_idx i];
    end
    if alpha2(i) ~= 0
       sv_idx = [sv_idx i];
    end 
    if alpha3(i) ~= 0
       sv_idx = [sv_idx i];
    end
end

y = [alpha1.*label1'*linear_kernel+bias1;  ...
    alpha2.*label2'*linear_kernel+bias2;  ...
    alpha3.*label3'*linear_kernel+bias3];

[colMax, colMaxIdx] = max(y);
correct = sum((trainLabel == colMaxIdx'));
train_accuracy = correct/trainTotal;

% plot 
xrange = [min(trainFeature(:,1))-move max(trainFeature(:,1))+move];
yrange = [min(trainFeature(:,2))-move max(trainFeature(:,2))+move];

inc = 0.01;

[gx, gy] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the 
% decision boundary image that is used as the plot background.
image_size = size(gx);
 
xy = [gx(:) gy(:)]; % make (x,y) pairs as a bunch of row vectors.

y1 = alpha1.*label1'*(xy*trainFeature')' + bias1;
y2 = alpha2.*label2'*(xy*trainFeature')' + bias2;
y3 = alpha3.*label3'*(xy*trainFeature')' + bias3;

y_new = [y1; y2; y3];
[colMax_new, colMaxIdx_new] = max(y_new);
decisionmap = reshape(colMaxIdx_new, image_size);

figure;
 
%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
 
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
colormap(cmap);
 
% plot the class training data.
plot(trainFeature(1:40, 1),trainFeature(1:40,2), 'rx');
plot(trainFeature(41:80,1),trainFeature(41:80,2), 'g+');
plot(trainFeature(81:120,1),trainFeature(81:120,2), 'b*');

% plot support vector
for i=1:120
    plot(trainFeature(sv_idx(i), 1),trainFeature(sv_idx(i),2), 'blacko');
end

% include legend
legend('Class 1', 'Class 2', 'Class 3','Support Vector','Location','NorthOutside', ...
    'Orientation', 'horizontal');
 
% label the axes.
xlabel('x1');
ylabel('x2');

%%%%%%%%%%% test stage %%%%%%%%%%%
linear_kernel = testFeature * trainFeature';

y = [alpha1.*label1'*linear_kernel'+bias1;  ...
    alpha2.*label2'*linear_kernel'+bias2;  ...
    alpha3.*label3'*linear_kernel'+bias3];

[colMax, colMaxIdx] = max(y);
correct = sum((testLabel == colMaxIdx'));
test_accuracy = correct/testTotal;
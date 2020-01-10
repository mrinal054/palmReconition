clc; clear all; close all;

format long

% Call dataset
trn = dlmread('HDCT250_2x2_train.txt');              % 
tst = dlmread('HDCT250_2x2_test.txt');               %

% Variables
H = 100;                                        % No. of hidden neurons
CL = 250;                                       % No. of classes
P = 1750;                                        % No. of train images
Pt =250;                                       % No. of test images
PA = 7;                                         % No. of images per class
PAT = 1;                                        % No. of test image per folder
NTP = Pt/PAT;                                   % No. of test persons

% Normalization


% Target
val = 1;
T1 = eye(CL);
for i = 1:CL
    T2(:,val:val+PA-1) = repmat(T1(:,i),1,PA);
    val = val+PA;
end
T = T2;                                         % column-wised distributed 100x700
clear val

% Network
net = patternnet(H);
net = train(net,trn,T);
view(net)
y = net(tst);                                   % net = train(net,x,t);
% perf = perform(net,T,y);
classes = vec2ind(y);

% Performance
error = 0;
success = 0;

val = 1;
for i = 1:NTP
    TP(val:val+PAT-1) = i;
    val = val + PAT;
end
clear val

for i = 1:length(classes)
    if classes(i) == TP(i)
        success = success+1;
    else
        error = error+1;
    end
end

%----------------------------------------------------
rate_percent=(success/(success+error))*100


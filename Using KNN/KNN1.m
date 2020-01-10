clc; clear all; close all;
%------------------------------------------
% Variable
N = 256;                  % No. of PCs
P = 700;                 % No. of train images (total)
Pt = 100;                % No. of test images
PA = 7;                  % No. of image per folder
TA = 100;                % No. of class
success = 0;
error = 0;

% Dataset
search_item_train = '*BDCT_train8*.txt';
set_folder_location = 'D:\academic\LU\project\Palm print detection\Code by Us\ROC';
[outTrain infoTrain] = readFile(search_item_train,set_folder_location);     % outTrain colm-wise; such as 256x700

search_item_test = '*BDCT_Test8*.txt';
[outTest infoTest] = readFile(search_item_test,set_folder_location);        % outTest colm-wise; such as 256x100

trn = outTrain{1};
tst = outTest{1};

train = trn';                             % 200x80
test = tst';                              % 200x80

% Group/Target
val = 1;
for i = 1:PA:P
    G(i:i+PA-1) = val;
    val = val + 1;
end

% KNN classifier
class = knnclassify(test, train, G)

% Counting
GT = 1:TA;
for i = 1:Pt
    if class(i) == GT(i)
        success = success + 1;
    else 
        error = error + 1;
    end
end

rec = (success/(success+error))*100

clc; clear all; close all;
%------------------------------------------
% Variable
N = 256;                 % No. of PCs
P = 1750;                 % No. of train images 
Pt = 250;                % No. of test images
PA = 7;                  % No. of image per folder
TA = 250;                % No. of class
success = 0;
error = 0;

%% Dataset
search_item_train = 'HDCT250_9x9_train.txt';
set_folder_location = 'C:\Users\Sanowar Hossain\Desktop\PALM.FINAL\Using KNN';
[outTrain infoTrain] = readFile(search_item_train,set_folder_location);     % outTrain colm-wise; such as 256x700

search_item_test = 'HDCT250_9x9_test.txt';
[outTest infoTest] = readFile(search_item_test,set_folder_location);        % outTest colm-wise; such as 256x100

trn = outTrain{1};
tst = outTest{1};

train = trn';                             % 200x80
test = tst';                              % 200x80
%% Find min distance
D = dist(test,trn);     % test row-wise, trn colm-wise
[minValue_D, cnt] = min(D');

%% Class determination
class1 = cnt./PA;
for i = 1:Pt
    if class1(i) < 0
        class(i) = 1;
    else
        class(i) = ceil(class1(i));
    end
end

%% Counting
for i = 1:Pt
    if i == class(i)
        success = success + 1;
    else 
        error = error + 1;
    end
end

%% Recognition rate
rec = (success/(success+error))*100

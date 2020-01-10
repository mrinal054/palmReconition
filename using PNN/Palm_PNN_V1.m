% Learning PNN technique
clc; clear all; close all;
%----------------------------------------
format long

% Call training and testing dataset
trn = dlmread('HDCT250_9x9_train.txt');         % 80x200
tst = dlmread('HDCT250_9x9_test.txt');          % 80x200
TA = 250;

% Variables
N = 64;                 % No. of PCs
P = 1750;
Pt = 200;               % No. of image (total)
PA = 7;                 % No. of image per folder
PAt = 1;
success = 0;
error = 0;

for i = 1:Pt            % Calling testing image            
               
        % Start Gaussian
        d1 = dist(tst(:,i)',trn);
        d2 = d1.^2;
               
        % sigma calculation
                
%         [BB,indx]=sort(d2,'ascend');%         
%         sigma = sqrt((BB(1)^2+BB(2)^2)/2);%         
%         sig =sigma;  %%size(sig)=(40,1)
%         sigSq = sig^2;
        
        sig2 = std2(trn);
        sig2Sq = sig2^2;
        
        % Distance Calculation
        a1 = -d2/(2*sig2Sq);            % version 02
        a2 = exp(a1);
%         b1 = sqrt((2*pi*sigSq).^N);
%         a3 = a2./b1
        a3 = a2./N;
        
        % Sum at output layer
        val = 1;
        for qq = 1:TA
            S1 = a3(val:val+PA-1);
            S2(qq) = sum(S1);
            val = val+PA;
        end
        
        % Class
        R1 = find(max(S2)==S2);
        R2(i) = R1(1);          % Need care 
        
        if R2(i) == i                   % THIS METHOD IF ONLY ONE IMAGE PER CLASS
            success = success + 1;
        else
            error = error + 1;
        end
        
%         if mod(i,PA) == 0
%             if R1 == i/PA
%                 success = success + 1;
%             else 
%                 error = error + 1;
%             end
%             
%         else mod(i,PA) ~= 0
%             if (floor(i/PA)+1) == R1
%                 success = success + 1;
%             else
%                 error = error + 1;
%             end
%         end 
        
end

%% Counting success
rate = (success/(success + error))*100

    
    
    
    
    
    

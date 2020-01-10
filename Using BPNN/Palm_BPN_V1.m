clear memory; clear all; clc
format long

nump=100;  % number of classes
 
n=7;     % number of images per class

% training images reshaped into columns in P 
% image size (3x3) reshaped to (1x9)


% training images 
trn=dlmread('BDCT_8_train.txt');
P = trn;
% testing images 
tst = dlmread('BDCT_8_test.txt');
N = tst;

% Normalization
P=P/max(max(P));
N=N/max(max(P));

% targets
val = 1;
T1 = eye(nump);
for i = 1:nump
    T2(:,val:val+n-1) = repmat(T1(:,i),1,n);
    val = val+n;
end
T = T2;                                    % column-wised distributed 100x700
clear val

S1=300;   % numbe of hidden layers
S2=100;   % number of output layers (= number of classes)

[R,Q]=size(P); 
epochs = 50;      % number of iterations
goal_err = 10e-5;    % goal error
a=0.3;                        % define the range of random variables
b=-0.3;
W1=a + (b-a) *rand(S1,R);     % Weights between Input and Hidden Neurons    5x700
W2=a + (b-a) *rand(S2,S1);    % Weights between Hidden and Output Neurons   100x5
b1=a + (b-a) *rand(S1,1);     % Weights between Input and Hidden Neurons    5x1
b2=a + (b-a) *rand(S2,1);     % Weights between Hidden and Output Neurons   100x1
n1=W1*P;                        % 5x64
A1=logsig(n1);                  % 5x64
n2=W2*A1;                       % 100x64
A2=logsig(n2);                  % 100x64
e=A2-T;
error =0.5* mean(mean(e.*e));    
nntwarn off
for  itr =1:epochs
    if error <= goal_err 
        break
    else
         for i=1:Q
            df1=dlogsig(n1,A1(:,i));
            df2=dlogsig(n2,A2(:,i));
            s2 = -2*diag(df2) * e(:,i);			       
            s1 = diag(df1)* W2'* s2;
            W2 = W2-0.1*s2*A1(:,i)';
            b2 = b2-0.1*s2;
            W1 = W1-0.1*s1*P(:,i)';
            b1 = b1-0.1*s1;

            A1(:,i)=logsig(W1*P(:,i),b1);
            A2(:,i)=logsig(W2*A1(:,i),b2);
         end
            e = T - A2;
            error =0.5*mean(mean(e.*e));
            disp(sprintf('Iteration :%5d        mse :%12.6f%',itr,error));
            mse(itr)=error;
    end
end

threshold=0.9;   % threshold of the system (higher threshold = more accuracy)

% training images result

%TrnOutput=real(A2)
TrnOutput=real(A2>threshold);    

% applying test images to NN
n1=W1*N;
A1=logsig(n1);
n2=W2*A1;
A2test=logsig(n2)

% % testing images result
% 
% %TstOutput=real(A2test)
% TstOutput=real(A2test>threshold)
% 
% 
% % recognition rate
% wrong=size(find(TstOutput-T),1);
% recognition_rate=100*(size(N,2)-wrong)/size(N,2)
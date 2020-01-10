clc; clear all; close all;
format long;
%---------------------------------------------
%variables
N=64;%no. of feature
H=300;%hidden neuron
TA=200; %no. of target node & no. of individual person
P=1400; %no. of pattern
Pt=200;
PA=7; %no. of persons per folder
PAt=1;
lr=.1; %learning rate
IT=500; %no. of iteration
%-------------------------------------------------
%target preparing
T=eye(TA);
v1=zeros(TA,P);
for i=1:TA
    for j=1:PA
          v1(:,(i-1)*PA+j)=T(:,i);
    end
 end
T1=v1;%%size(T1)=(40,200)
T2=T1'; %%size(T2)=(200,40)
%----------------------------------------------------------
%for input image(training)
trn = dlmread('BDCT200_8_train.txt');           % 80x200 % Column-wise data 
s_input = trn(:,1:P)'; 
[roww,clm]=size(s_input);
s_train=s_input';
%------------------------------------------------------------
% %for testing
% %feature for test data                 
% p=y-uint8(single(m)*single(O));   % vzm is v with the mean removed. 
% %p=r-m; 
tst = dlmread('BDCT200_8_test.txt');            % 80x200
s_test = tst(:,1:Pt)';


% s_test=single(single(p))'*feature_matrix; %size(s_test)=(200,50)
% test=double(s_test');
%---------------------------------------------------------------
%center
[index,c]=kmeans(s_input,H);
%-------------------------------------------------------------
%euclidian dist calculation
for j=1:P 
    for i=1:H
    d1=s_input(j,:)-c(i,:);
    d2=d1.*d1;%without root
    dis(j,i)=sum(d2);
    end
end
dist=dis;    %%size(dist)=(400,40)
%distacne=sqrt(dist)
%-------------------------------------------------------------
% sigma calculation
% % METHOD 1
% [BB,indx]=sort(sqrt(dist),'ascend');
% for i=1:H
%         sigma(i,:)= sqrt((BB(1,i)^2+BB(2,i)^2)/2);
% end
% sig=sigma;  %%size(sig)=(40,1)

% METHOD 2
sig2 = std2(s_input);
sigma = sig2;

% % METHOD 3
% BB1 = sum(dist)./(P-1);
% sigma = BB1';


%-------------------------------------------------------------
%gaussian rbf calculation
for j=1:P
    for i=1:H
        g(j,i)=exp(-dist(j,i)/(2* sigma^2));
    end
end
G=g;
[row,coloum]=size(G); %%size(G)=(400,40) %%1st column means G1
%------------------------------------------------------------
%adding bias for training
for j=1:row 
    mm(j,:)=1;
end
b1=mm;
Gb=[b1 G]; %%size(Gb)=(400,41)

%euclidian dist calculation (test)
for j=1:Pt 
    for i=1:H
    d1_test=s_test(j,:)-c(i,:);
    d2_test=d1_test.*d1_test;%without root
    dis_test(j,i)=sum(d2_test);
    end
end
dist_test=dis_test;    %%size(dist)=(400,40)

%gaussian rbf calculation (test)
for j=1:Pt
    for i=1:H
        g_test(j,i)=exp(-dist_test(j,i)/(2* sigma^2));
    end
end
G_test=g_test;
[rowt,coloumt]=size(G_test); %%size(G)=(400,40) %%1st column means G1
%------------------------------------------------------------
%adding bias for testing
for j=1:rowt 
    mmt(j,:)=1;
end
b1t=mmt;
Gb_test=[b1t G_test]; %%size(Gb)=(400,41)
%------------------------------------------------------------
%random weight
% w1=rand(TA,H+1);
% w2=2*w1;
% W=w2-1; %%size(W)=(40,41)

aW=0.3;                        % define the range of random variables
bW=-0.3;
W=aW + (bW-aW) *rand(TA,H+1);
%----------------------------------------------------------
%weight update
for b=1:1:IT %%150 iteration
    gg(b)=b;
    temp=0;
    temp_test=0;
    for e=1:1:P %400 pattern %%changable
        h=Gb(e,:);
%         h_test=Gb_test(e,:);         %extra add
        Y=Gb(e,:)*W'; %size(Y)=(1,40) %%%%%%need care 
                      
  
%         Y_test=Gb_test(e,:)*W';      %extra add
%         YY(:,e)=Y_test;
        T3=T2(e,:);        
        for c=1:1:TA
            for d=1:1:H+1
                del_W(c,d)=lr*(T3(c)-Y(c))*h(d); %size(del_W=(40,41))                
            end         
        end
        W=W+del_W;
        error=0;
%         error_test=0;
        for g=1:1:TA %no. of output node %%changable
            error=error+(T3(g)-Y(g))^2;
%             error_test=error_test+(T3(g)-Y_test(g))^2; %for test
        end
        error=error/TA;
%         error_test=error_test/TA;           %for test
        temp=temp+error;
%         temp_test=temp_test+error_test;     %for test
           
    end
    error=(temp/(2*P));
%     error_test=(temp_test/(2*P));           %for test
    ee(b)=error;   
%     ee_test(b)=error_test;                  %for test
end
%-----------------------------------------------------------------
YY=Gb_test*W';      %extra add



%plotting
figure; plot(gg,ee); xlabel('iteration'); ylabel('mse')
% figure;
% plot(gg,ee_test)
% xlabel('iteration');ylabel('mse')
%axis([1 150 0 .01])
ee(IT)
ax=YY; %%size(ax)=[40,200]
%----------------------------------------------------------------
%performance counting
[aa,nn]=max(ax)
success = 0;
err = 0;
for i = 1:TA
    if nn(i) == i;
        success = success + 1;
    else
        err = err + 1;
    end
end
rate = (success/(success + err))*100
% FD=w(:,nn);
% %-------------------------------------------------------------------
% %winner take all
% for k=1:Pt      %200
%     for l=1:TA   %40
%         if ax(l,k)==aa(k) 
%             out(l,k)=1;
%         else
%             out(l,k)=0;
%         end 
%     end
% end
% output=out;

%-----------------------------------------------------------------
        
 dlmwrite('center_RBFN_BDCT16.txt',c,'delimiter','\t','newline','pc') 
 dlmwrite('weight_RBFN_BDCT16.txt',W,'delimiter','\t','newline','pc') 
        

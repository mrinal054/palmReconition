% palm 07
%ROI+DCT
% function out=database03();
clear all; clc; close all;   

PA = 7;                     % No. of training images per folder

 for i=1:250 %40
        %cd(strcat('s',num2str(i)));
        for j=1:PA %7
%--------------------------------------------------------------------------
            %image calling            
                    path_NO=i;
                    im_name =j;  
                    im_path='C:\Users\Sanowar Hossain\Desktop\PALM.FINAL\Database3\s';
                    a=newroi(im_path,path_NO,im_name);          % UDF for ROI
             %image is reshaped to a square matrix
%--------------------------------------------------------------------------
%   dct compression            
dct_im=dct2(a);
            dct_im=(dct_im);            % neglecting absolute values
           for i1=1:128
               for j1=1:128
                   if (i1+j1)>80
                       dct_im(i1,j1)=0;
                   end
               end
           end
%--------------------------------------------------------------------------

rowVal = 2;
colVal01 = 2;
intVal02 = 1;
        for ii = 1:rowVal
            for jj = 1:colVal01
                idct_im(intVal02)=dct_im(ii,jj);
                intVal02 = intVal02 + 1;
            end            
            colVal01 = colVal01 - 1;
        end
            v(:,(i-1)*7+j)=idct_im(:);          
    end        
end

dlmwrite('HDCT250_2x2_train.txt',v,'delimiter','\t','newline','pc')
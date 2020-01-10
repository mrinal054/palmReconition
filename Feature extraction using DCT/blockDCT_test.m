% ROI+DCT
clear all; clc; close all;

PAt = 1;                      % No. of image per folder

 for i=1:250 %100       
        for j=1:PAt %7
%--------------------------------------------------------------------------
            %image calling            
                    path_NO=i;
                    im_name =j;  
                    im_path='C:\Users\Sanowar Hossain\Desktop\PALM.FINAL\Database3\t';
                    a=newroi(im_path,path_NO,im_name);          % UDF for ROI
%                     a = imnoise(a,'gaussian',0.02,0.02);
                    %image is reshaped to a square matrix
%--------------------------------------------------------------------------
        % Make blocks

        cellSize = 64;       % Indicates cell size: 8x8
        cellNo = size(a,1)/cellSize;     % Indicates no. of cell in x and y axis
        cellRow = cellSize*ones(1,cellNo);   % [8 8 8 8 8.... upto 16 values]
        cellColm = cellSize*ones(1,cellNo);   % [8 8 8 8 8.... upto 16 values]
        % Divide matrix into predefined cells
        cellMat01 = mat2cell(a, cellRow, cellColm); % 16x16
        cellMat02 = cellMat01(:); % Making it colm wise 1D matrix
        % Perform DCT in each cell
        val = 1;
            for k = 1:size(cellMat02,1)
                I1 = dct2(cellMat02{k});
                I2(val) = I1(1);
                val = val + 1;
            end
        idct_im = I2; 
        v(:,(i-1)*PAt+j)=idct_im(:);          
        end
 end

dlmwrite('BDCT250_64_test.txt',v,'delimiter','\t','newline','pc')
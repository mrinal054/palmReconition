function out=load_database_noise2();
% We load the database the first time we run the program.

persistent loaded;
persistent wz;

if(isempty(loaded))
    v1=zeros(10304,200);
    v2=zeros(10304,200);
    for i=1:40
        cd(strcat('s',num2str(i)));
        for j=1:5
            a=imread(strcat(num2str(j),'.pgm'));
            b=imnoise(a,'gaussian',0,0.002);
          
            v1(:,(i-1)*5+j)=reshape(a,size(b,1)*size(a,2),1);
            v2(:,(i-1)*5+j)=reshape(b,size(b,1)*size(b,2),1);            
        end
        cd ..
    end
    wz=[uint8(v1) uint8(v2)]; % Convert to unsigned 8 bit numbers to save memory. 
end
loaded=1;  % Set 'loaded' to aviod loading the database again. 
out=wz;




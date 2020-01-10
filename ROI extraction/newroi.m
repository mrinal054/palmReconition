% palm 03
function ROIout = newroi(impath, pathNO, imname)

% pathNO is for folder number; imname is for image number;

A = imread([impath, num2str(pathNO),'\',num2str(imname),'.jpg']);

% figure;imshow(A); title('Original Image')
G = fspecial('gaussian',[5 5], 2);      % Applying the Smoothing Filter
C = imfilter(A, G, 'same');
[w, h] = size(C);

BW = im2bw(C, graythresh(C));           % Convert output into Binary
[B,~,~] = bwboundaries(BW);             % Detect the Boundary

centroid = regionprops(BW,'Centroid');  % Locate the centroid
centroid = centroid.Centroid;
% Cendroid(1) - X coordinate of Centroid
% Cendroid(2) - Y coordinate of Centroid

outline = flipud(B{1});     % This is the edge we are interested in

regionLine = zeros(length(outline),3);  % Put all coordinates and distances in the same array

for i = 1:length(outline)
    regionLine(i,:) = [outline(i,2) outline(i,1) sqrt((outline(i,2)-centroid(1))^2+(outline(i,1)-centroid(2))^2)];
end

% Use -regionLine (flip the line up side down) to find minima...
[~, i] = findpeaks(smooth(-regionLine(:,3),50),'MINPEAKDISTANCE',10,'MINPEAKHEIGHT',mean(-regionLine(:,3)));

coordinates = [regionLine(i(1:3),1:2); regionLine(i(end),1:2)];


sortCoord = sortrows(coordinates,2);    % Sort the coordinates from top of image to bottom
                                        % Descending order
%function score = lineScore(p)
p=sortCoord;


%%

d1 = sqrt((p(2,2)-p(1,2))^2 + (p(2,1)-p(1,1))^2);
d2 = sqrt((p(4,2)-p(3,2))^2 + (p(4,1)-p(3,1))^2);
sc = [d1 d2];

%end

%sc = lineScore(sortCoord);              % Distance based score to reject one point
if sc(1)<sc(2)
    sortCoord(4,:) = [];                % Reject point 4 (1, 2, 3 are close)
else
    sortCoord(1,:) = [];                % Reject point 1 (2, 3, 4 are close)
end
%%
%% Uncomment the following codes if you want to show four points on palm
% Mark the four points
% figure; imshow(A); title('Set point of interest')
% hold on
% plot(sortCoord(1,1),sortCoord(1,2),'o','linewidth',2,'MarkerEdgeColor','g','MarkerSize',8,'MarkerFaceColor','m');
% text(sortCoord(1,1)-30,sortCoord(1,2)-15,'P1','color','cyan','FontSize',12)
% plot(sortCoord(2,1),sortCoord(2,2),'o','linewidth',2,'MarkerEdgeColor','g','MarkerSize',8,'MarkerFaceColor','m');
% text(sortCoord(2,1)-30,sortCoord(2,2)-15,'P2','color','cyan','FontSize',12)
% plot(sortCoord(3,1),sortCoord(3,2),'o','linewidth',2,'MarkerEdgeColor','g','MarkerSize',8,'MarkerFaceColor','m');
% text(sortCoord(3,1)-40,sortCoord(3,2)-15,'P3','color','cyan','FontSize',12)

Cor01 = [sortCoord(1,1) sortCoord(2,1)];
Cor02 = [sortCoord(1,2) sortCoord(2,2)];
Cor03 = [sortCoord(2,1) sortCoord(3,1)];
Cor04 = [sortCoord(2,2) sortCoord(3,2)];

% Uncomment the following codes to draw lines
% line(Cor01,Cor02,'linewidth',4);
% line(Cor03,Cor04,'linewidth',4);
% plot(centroid(1),centroid(2),'r*')

%%
% Now the target is to extract a square of palm. For this I need four
% points. Define them: x2,y2 (upper left), x3,y3 (bottom left), x4,y4 (upper
% right), x5,y5 (bottom right)

% Measure distance between 1st and 2nd points
ds1 = sqrt((sortCoord(1,1) - sortCoord(2,1))^2 + (sortCoord(1,2) - sortCoord(2,2))^2);
thetaRad01 = atan((sortCoord(2,2)-sortCoord(1,2))/(sortCoord(2,1)-sortCoord(1,1)));
thetaRad02 =  - thetaRad01;
[x2 y2] = corVal(sortCoord(1,1), sortCoord(1,2),0.5*ds1, thetaRad02);
% plot(x2,y2,'r*') % x2 & y2 are two extended points; extended by half of ds1; upper left cooridnates  

% Measure distance between 2nd and 3rd points
ds2 = sqrt((sortCoord(3,1) - sortCoord(2,1))^2 + (sortCoord(3,2) - sortCoord(2,2))^2);
thetaRad03 = atan((sortCoord(2,2)-sortCoord(3,2))/(sortCoord(2,1)-sortCoord(3,1)));
thetaRad04 =  thetaRad03;
[x3 y3] = corVal(sortCoord(3,1), sortCoord(3,2),0.5*ds2, thetaRad04);
% plot(x3,y3,'b*') % % x3 & y3 are two extended points; extended by half of ds2; lower left cooridnates  

% L1 = line([x3 x2], [y3 y2], 'linewidth',4, 'color', 'black');

% Measure distance between two extended points
ds3 = sqrt((x3 - x2)^2 + (y3 - y2)^2); % distance between two extended points
ds3 = round(ds3);

% Calculate slope of line between (x2,y2) & (x3,y3)
M = (y3 - y2)/(x3 - x2);
newM = - 1/M; % slope of perpendicular line: m1*m2 = -1

% Calculate angle for x2,y2 & x3,y3
thetaRad05 = atan(newM);  % calculate angle of the tangent
[x4 y4] = corVal(x2,y2,ds3,thetaRad05); % upper right coordinates of the square
% L2 = line([x2 x4],[y2 y4], 'linewidth',4,'color','red');

[x5 y5] = corVal(x3,y3,ds3,thetaRad05); % lower right coordinates of the square
% L3 = line([x3 x5],[y3 y5], 'linewidth',4,'color','blue');
% L4 = line([x4 x5],[y4 y5],'linewidth',4,'color','green');


%%
%Rotate image to make it vertical
thetaDeg06 = (atan((y5-y3)/(x5-x3)))*180/pi; % Angle in degree

% First rotate points
% To do so, make points as image
zeroXY2 = zeros(w,h);
zeroXY3 = zeros(w,h);
zeroXY4 = zeros(w,h);
zeroXY5 = zeros(w,h);

% Place the points in a black image
zeroXY2(round(y2)-1:round(y2)+1,round(x2)-1:round(x2)+1) = 255; %figure; imshow(zeroXY2);
zeroXY3(round(y3)-1:round(y3)+1,round(x3)-1:round(x3)+1) = 255; %figure; imshow(zeroXY3);
zeroXY4(round(y4)-1:round(y4)+1,round(x4)-1:round(x4)+1) = 255;
zeroXY5(round(y5)-1:round(y5)+1,round(x5)-1:round(x5)+1) = 255;

% Rotate the images
ImzeroXY2 = imrotate(zeroXY2, thetaDeg06); %figure; imshow(ImzeroXY2);
ImzeroXY3 = imrotate(zeroXY3, thetaDeg06); %figure; imshow(ImzeroXY3);
ImzeroXY4 = imrotate(zeroXY4, thetaDeg06);
ImzeroXY5 = imrotate(zeroXY5, thetaDeg06);

% Find the white pixels in rotated images
[rx2,ry2] = find(ImzeroXY2 == 255);  % r means rotated
[rx3,ry3] = find(ImzeroXY3 == 255);
[rx4,ry4] = find(ImzeroXY4 == 255); %figure; imshow(ImzeroXY4)
[rx5,ry5] = find(ImzeroXY5 == 255); %figure; imshow(ImzeroXY5)
Im2 = imrotate(A, thetaDeg06);

%% Uncomment the following codes if you want to draw lines and points on palm

% figure; imshow(Im2); hold on; title('Rotated Image')

% Draw lines
% line([ry3(1) ry2(1)],[rx3(1) rx2(1)], 'linewidth',4,'color','yellow'); % (1) has been added for safety
% line([ry2(1) ry4(1)],[rx2(1) rx4(1)], 'linewidth',4,'color','yellow');
% line([ry4(1) ry5(1)],[rx4(1) rx5(1)], 'linewidth',4,'color','yellow');
% line([ry3(1) ry5(1)],[rx3(1) rx5(1)], 'linewidth',4,'color','yellow');

% Mark corners
% plot(ry2(1),rx2(1),'o','linewidth',2,'MarkerEdgeColor','k','MarkerSize',8,'MarkerFaceColor','m')
% plot(ry3(1),rx3(1),'o','linewidth',2,'MarkerEdgeColor','k','MarkerSize',8,'MarkerFaceColor','m')
% plot(ry4(1),rx4(1),'o','linewidth',2,'MarkerEdgeColor','k','MarkerSize',8,'MarkerFaceColor','m')
% plot(ry5(1),rx5(1),'o','linewidth',2,'MarkerEdgeColor','k','MarkerSize',8,'MarkerFaceColor','m')
% hold off

Im3 = Im2(rx2:rx2+ds3,ry2:ry2+ds3); 
% figure; 
% subplot(2,2,1);imshow(Im3); title('size: 249x249') % size: 249x249
ROIout = imresize(Im3,[128 128]); 
% subplot(2,2,2); imshow(Im4); title('size: 128x128')

%Lap = fspecial('laplacian', .2);      % Applying the Laplace filter
%Im5 = imfilter(Im4, Lap, 'same');
% subplot(2,2,3); imshow(Im5); title('Image filtered by HPF');
%ROIout = histeq(Im5);
% subplot(2,2,4); imshow(Im6); title('After histogram equilizing')
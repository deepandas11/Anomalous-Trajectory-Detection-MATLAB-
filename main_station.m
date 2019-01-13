clear all, clc
clear classes;
tracker = MultiObjectTrackerKLT;
videoname = 'station_original.avi'  %put the path to the video (.mp4) file
videoFileReader = vision.VideoFileReader(videoname);
videoPlayer = vision.VideoPlayer('Position', [100, 100, 720, 480]);
objectFrame = step(videoFileReader);
frame  = rgb2gray(objectFrame);

%% Tracking %% 
tic;
sz = size(frame);
height = sz(1); width = sz(2);
patchSize = 16;
X= floor(width/patchSize); Y = floor(height/patchSize);
disp('Creating boxes');
M=0;
% dividing the image frame into blocks and 
% initializing the x and y coordinates of each trajectory
for i = 1: X
    for j = 1: Y
        M = M+1;
        boxes(M,1) = (i-1)*patchSize + 1;    
        boxes(M,2) = (j-1)*patchSize + 1;
        boxes(M,3) = patchSize;
        boxes(M,4) = patchSize;
        trajectory(1,2*M-1) = boxes(M,1)+patchSize/2;
        trajectory(1,2*M) = boxes(M,2)+patchSize/2;
    end
end
disp('Adding boxes to tracker');
tracker.addDetections(frame, boxes);
frameNumber = 1;
keepRunning = true;
disp('Press Ctrl-C to exit...');
frameRefreshRate = 40; % frames after which system adds new blocks to the tracker
while ~isDone(videoFileReader)
    % updating the x and y coordinates of each keypoint after each frame
    frameNumber = frameNumber + 1
    framergb = step(videoFileReader);
    frame = rgb2gray(framergb);     
    if mod(frameNumber, frameRefreshRate) == 0
            prior  = size(tracker.Bboxes);
            tracker.addDetections(frame, boxes);
            after = size(tracker.Bboxes);    
            fprintf( ' Added %d boxes \n', (after(1)-prior(1)));
    else
        tracker.track(frame);
    end
    
    displayFrame = insertObjectAnnotation(frame, 'rectangle',tracker.Bboxes, tracker.BoxIds);
    displayFrame = insertMarker(displayFrame, tracker.Points);
    videoPlayer.step(displayFrame);
    sz = size(tracker.BoxIds);
    for i = 1:sz(1)
        M =tracker.BoxIds(i);
        x = tracker.Bboxes(i,1) + tracker.Bboxes(i,3)/2;
        y = tracker.Bboxes(i,2) + tracker.Bboxes(i,4)/2;
        trajectory(frameNumber, 2*M-1) = x;
        trajectory(frameNumber, 2*M) = y;
    end
   
end
release(videoPlayer);
toc;


%% Calculating Trajectory after tracking boxes %% 


clearvars trajec_clean trajec_feat;
figure(1);
hold 'on'
counter = 0;
lenthresh =50; % threshold for minimum length of valid trajectory
col = rand(1,3);
for i =1:M
    x = trajectory(:,2*i-1);
    y = trajectory(:,2*i);
    check = find(x ~= 0);
    X = trajectory(check,2*i-1);
    Y = trajectory(check,2*i);
    deltax = max(X) - min(X); 
    deltay = max(Y) - min(Y);
    length(i) = (deltax^2 + deltay^2)^(1/2);
    if (length(i) > lenthresh)
        counter = counter + 1;
        Y = -1*Y;
        trajec_clean(:,2*counter-1) = x;
        trajec_clean(:,2*counter) = y;
        plot(X,Y,'b:');
        plot(X(1), Y(1), 'g.');
        %plot(X(1), Y(1), 'bsquare');
    end
end
axis([0 width  -height 0]);
hold on;


%% Parameterizing the trajectories

disp('Parametizing the trajectories')
[frames , numTrajec] =  size(trajec_clean);
count_trajec = numTrajec;
numTrajec = numTrajec/2;
for i = 1 : numTrajec
    x = trajec_clean(:,2*i - 1);
    y = -1*trajec_clean(:,2*i);
    check = find(x ~= 0);
    x = x(check);
    y = y(check);
    [sz extra] = size(x);
    t = 1:sz;
    t = t';
    paramx = polyfit(t,x,3);
    paramy = polyfit(t,y,3); 
    meanx = mean(x);
    meany = mean(y);
    x1 = x(1:end-1,:);
    x2 = x(2:end,:);
    y1 = y(1:end-1,:);
    y2 = y(2:end,:);
    deltax = x2-x1;
    deltay = y2-y1;
    angle = floor(atan2(deltay,deltax)/pi * 4 + 4); % quantizing the angle
    check = angle == 8;
    angle (check ) = 0;  % now the angles between  0 to 7
    repAngle = median(angle);
    trajec_feat(i,:) = [paramx paramy  repAngle meanx meany  0 0 0 sz ];
end
%%
%deciding the dominant direction of motion among the trajectories
t_angle = trajec_feat(:,9);
check = find(t_angle == 0);
[a0 ex] = size(check);
check = find(t_angle == 1);
[a1 ex] = size(check);
check = find(t_angle == 2);
[a2 ex] = size(check);
check = find(t_angle == 3);
[a3 ex] = size(check);
check = find(t_angle == 4);
[a4 ex] = size(check);
check = find(t_angle == 5);
[a5 ex] = size(check);
check = find(t_angle == 6);
[a6, ex] = size(check);
check = find(t_angle == 7);
[a7 ex] = size(check);
hor = a0 + a3 + a4 + a7;
ver = a1 + a2 + a5 + a6;
trajec_feat(:,9) = 200;
if(hor >= ver)
    check = t_angle == 0 | t_angle == 1 | t_angle == 6 | t_angle == 7;
    trajec_feat(check,9) = -200;
else
    check = t_angle == 0 | t_angle == 1 | t_angle == 2 | t_angle == 3;
    trajec_feat(check,9) = -200;
end

%%
% calculating the density features for each trajectory
pairwise_dist = pdist2(trajec_feat, trajec_feat);
epsilon = 20;
check = (pairwise_dist <= epsilon);
k = sum(check,2);
trajec_feat(:,12) = k;

epsilon = 25;
check = (pairwise_dist <= epsilon);
k = sum(check,2);
trajec_feat(:,13) = k;

epsilon = 30;
check = (pairwise_dist <= epsilon);
k = sum(check,2);
trajec_feat(:,14) = k;

%%
%Standard Deviation of the trajectory points 
stdx = zeros(1,numTrajec);
stdy = zeros(1,numTrajec);
std_clean = std(trajec_clean, 0,1);
j=0;
for j = 1:numTrajec
    stdx(j) = std_clean((2*j)-1);
    stdy(j) = std_clean(2*j);
end

for i = 1:numTrajec
    trajec_feat(i,16) = stdx(i);
    trajec_feat(i,17) = stdy(i);
end


%%
%Spatio-temporal density 
distvec = zeros(1,numTrajec);
LocVec = zeros(frameNumber, numTrajec);
for i = 1:numTrajec
    for j = 1:frameNumber
        LocVec(j,i) = sqrt((trajec_clean(j, (2*i)-1))^2 + (trajec_clean(j, 2*i))^2);
    end
end
diff1 = zeros(frameNumber, numTrajec);
for i = 1:numTrajec
    for j = 1:numTrajec
        diff1(:,i) = diff1(:,i)+ abs(LocVec(:,i)-LocVec(:,j));
    end
end
diff1 = diff1/10^5;
diffvec = zeros(1, numTrajec);
for i = 1:numTrajec
    diffvec(i) = sum(diff1(:,i));
end
c = diffvec/1000;
for i = 1:numTrajec
    trajec_feat(i,18) = diffvec(i);
end

mean_diffvec = mean(diffvec);
mean_dv = repmat(mean_diffvec, 1, numTrajec);

%Plotting Spatio-temporal density
ts = 1:numTrajec;
figure(2);
plot(ts,diffvec,'g');
hold on;
plot(ts, mean_dv, 'r');
title('Accumulated Spatio-temporal Distance');
xlabel('Trajectory');
ylabel('Distance');

%%
%Normalizing Feature 
siz1 = size(trajec_feat);
nfeat = zeros(siz1);
for i=1:siz1(2)
    a = trajec_feat(:,i);
    norma = (a - min(a))/(max(a)-min(a));
    nfeat(:,i) = norma;
end

%%
VoteVec = zeros(1,numTrajec);


tic;
density = zeros(numTrajec, 3);
density(:,1:3) = trajec_feat(:, 12:14);
%density(:,4) = nfeat(:,18);
bandwidthDensity = 1.4;
[clustCentDensity,point2clusterDensity,clustMembsCellDensity] = HGMeanShiftCluster(density',bandwidthDensity,'gaussian');

m1= size(point2clusterDensity,2);
k= max(point2clusterDensity);
Colors1 = hsv(k);
figure(3);
h1 = axes;
title(['Clustering for Density with Bandwidth : ' num2str(bandwidthDensity) ' and Number of clusters : ' num2str(k)]);
hold on;
for i = 1:m1
    co1 = Colors1(point2clusterDensity(i),:);
    dfx = trajec_clean(:,2*i - 1);
    dfy = trajec_clean(:,2*i);
    check = find(dfx ~=0 & dfy ~=0);
    Dfx = trajec_clean(check, 2*i - 1);
    Dfy = trajec_clean(check, 2*i);
    plot(Dfx, Dfy,  'Color',co1);
    plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co1);
    hold on;
end 
set(h1, 'Ydir', 'reverse');

nclustersdensity = size(clustCentDensity,2);
lambdaD = 0.766;
[HDensity,threshDensity,lambdaDensity] = ShannonAnomaly(density, nclustersdensity,clustCentDensity,lambdaD);
figure(4);
title(['Lambda  = ', num2str(lambdaDensity)]);
h2 = axes;
hold on;
for i = 1:m1
    if(HDensity(i)>threshDensity)
        VoteVec(i) = VoteVec(i)+1;
        co2 = Colors1(point2clusterDensity(i),:);
        dfx = trajec_clean(:,2*i - 1);
        dfy = trajec_clean(:,2*i);
        check = find(dfx ~=0 & dfy ~= 0);
        Dfx = trajec_clean(check, 2*i - 1);
        Dfy = trajec_clean(check, 2*i);
        plot(Dfx, Dfy, 'Color', co2);
        plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co2);
        hold on;
    end
end
set(h2, 'YDir', 'reverse');




shape = zeros(numTrajec, 8);
shape = trajec_feat(:, 1:8);
bandwidthShape = 100;
[clustCentShape,point2clusterShape,clustMembsCellShape] = HGMeanShiftCluster(shape',bandwidthShape,'gaussian');
m1= size(point2clusterShape,2);
k= max(point2clusterShape);
Colors2 = hsv(k);
figure(5);
h1 = axes;
title(['Clustering for Shape, Bandwidth : ' num2str(bandwidthShape) ' and Number of clusters : ' num2str(k)]);
hold on;
for i = 1:m1
    co3 = Colors2(point2clusterShape(i),:);
    dfx = trajec_clean(:,2*i - 1);
    dfy = trajec_clean(:,2*i);
    check = find(dfx ~=0 & dfy ~=0);
    Dfx = trajec_clean(check, 2*i - 1);
    Dfy = trajec_clean(check, 2*i);
    plot(Dfx, Dfy,  'Color',co3);
    plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co3);
    hold on;
end
set(h1, 'Ydir', 'reverse');

nclustersshape = size(clustCentShape,2);
lambdaS = 0.77;
[HShape,threshShape,lambdaShape] = ShannonAnomaly(shape, nclustersshape,clustCentShape, lambdaS);
figure(6);
title(['Lambda  = ', num2str(lambdaShape)]);
h2 = axes;
hold on;
for i = 1:m1
    if(HShape(i)>threshShape)
        VoteVec(i) = VoteVec(i)+1;
        co4 = Colors2(point2clusterShape(i),:);
        dfx = trajec_clean(:,2*i - 1);
        dfy = trajec_clean(:,2*i);
        check = find(dfx ~=0 & dfy ~= 0);
        Dfx = trajec_clean(check, 2*i - 1);
        Dfy = trajec_clean(check, 2*i);
        plot(Dfx, Dfy, 'Color', co4);
        plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co4);
        hold on;
    end
end
set(h2, 'YDir', 'reverse');


std = zeros(numTrajec, 2);
std = trajec_feat(:, 16:17);
bandwidthstd = 50;
[clustCentSTD,point2clusterSTD,clustMembsCellSTD] = HGMeanShiftCluster(std',bandwidthstd,'gaussian');
m1= size(point2clusterSTD,2);
k= max(point2clusterSTD);
Colors3 = hsv(k);
figure(7);
h1 = axes;
title(['Clustering for Standard Deviation, Bandwidth : ' num2str(bandwidthstd) ' and Number of clusters : ' num2str(k)]);
hold on;
for i = 1:m1
    co5 = Colors3(point2clusterSTD(i),:);
    dfx = trajec_clean(:,2*i - 1);
    dfy = trajec_clean(:,2*i);
    check = find(dfx ~=0 & dfy ~=0);
    Dfx = trajec_clean(check, 2*i - 1);
    Dfy = trajec_clean(check, 2*i);
    plot(Dfx, Dfy,  'Color',co5);
    plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co5);
    hold on;
end
set(h1, 'Ydir', 'reverse');

nclusterstd = size(clustCentSTD,2);
lambdastd = 0.73;
[HSTD,threshSTD,lambdaSTD] = ShannonAnomaly(std, nclusterstd,clustCentSTD, lambdastd);
figure(8);
title(['Lambda  = ', num2str(lambdaSTD)]);
h2 = axes;
hold on;
for i = 1:m1
    if(HSTD(i)>threshSTD)
        VoteVec(i) = VoteVec(i)+1;
        co6 = Colors3(point2clusterSTD(i),:);
        dfx = trajec_clean(:,2*i - 1);
        dfy = trajec_clean(:,2*i);
        check = find(dfx ~=0 & dfy ~= 0);
        Dfx = trajec_clean(check, 2*i - 1);
        Dfy = trajec_clean(check, 2*i);
        plot(Dfx, Dfy, 'Color', co6);
        plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co6);
        hold on;
    end
end
set(h2, 'YDir', 'reverse');



position = zeros(numTrajec, 3);
position = trajec_feat(:, 9:11);
bandwidthPosition = 120;
[clustCentPosition,point2clusterPosition,clustMembsCellPosition] = HGMeanShiftCluster(position',bandwidthPosition,'gaussian');
m1= size(point2clusterPosition,2);
k= max(point2clusterPosition);
Colors4 = hsv(k);
figure(9);
h1 = axes;
title(['Clustering for Position, Bandwidth : ' num2str(bandwidthPosition) ' and Number of clusters : ' num2str(k)]);
hold on;
for i = 1:m1
    co7 = Colors4(point2clusterPosition(i),:);
    dfx = trajec_clean(:,2*i - 1);
    dfy = trajec_clean(:,2*i);
    check = find(dfx ~=0 & dfy ~=0);
    Dfx = trajec_clean(check, 2*i - 1);
    Dfy = trajec_clean(check, 2*i);
    plot(Dfx, Dfy,  'Color',co7);
    plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co7);
    hold on;
end
set(h1, 'Ydir', 'reverse');

nclusterpos = size(clustCentPosition,2);
lambdapos = 0.78;
[HPos,threshPos,lambdaPos] = ShannonAnomaly(position, nclusterpos,clustCentPosition, lambdapos);
figure(10);
title(['Lambda  = ', num2str(lambdaPos)]);
h2 = axes;
hold on;
for i = 1:m1
    if(HPos(i)>threshPos)
        VoteVec(i) = VoteVec(i)+1;
        co10 = Colors4(point2clusterPosition(i),:);
        dfx = trajec_clean(:,2*i - 1);
        dfy = trajec_clean(:,2*i);
        check = find(dfx ~=0 & dfy ~= 0);
        Dfx = trajec_clean(check, 2*i - 1);
        Dfy = trajec_clean(check, 2*i);
        plot(Dfx, Dfy, 'Color', co10);
        plot(Dfx(1), Dfy(1), 'rsquare', 'Color',co10);
        hold on;
    end
end
set(h2, 'YDir', 'reverse');



figure(11);
len = sum(VoteVec);
AnomIndex = zeros(1,len);
title('Anomalous Trajectories');
h3 = axes;
hold on;
c= 0;
for i = 1:numTrajec
   if(VoteVec(i)>2)
       c = c+1;
       dfx = trajec_clean(:,2*i - 1);
       dfy = trajec_clean(:,2*i);
       check = find(dfx ~=0 & dfy ~= 0);
       Dfx = trajec_clean(check, 2*i - 1);
       Dfy = trajec_clean(check, 2*i);
       plot(Dfx, Dfy,'r');
       plot(Dfx(1), Dfy(1), 'b.');
       %txt1 = ['\leftarrow Trajec - ', num2str(i)];
       %text(Dfx(5), Dfy(5), num2str(i));
       %disp(['Trajectory - ', num2str(i)]);
       hold on;
   end
end
set(h3, 'YDir', 'reverse');
toc;

%%
ObsAnom = zeros(1,numTrajec);
for i = 1:numTrajec
    if(VoteVec(i) >2)
        ObsAnom(i) = 1;
    end
end

figure(12);
hold on;
h4 = axes;
for i = 1:m1
    dfx = trajec_clean(:,2*i - 1);
    dfy = trajec_clean(:,2*i);
    check = find(dfx ~=0 & dfy ~=0);
    Dfx = trajec_clean(check, 2*i - 1);
    Dfy = trajec_clean(check, 2*i);
    if(ObsAnom(i) == 0)
        plot(Dfx, Dfy, 'b:');
        plot(Dfx(1), Dfy(1), 'ro');
        text(Dfx(5), Dfy(5), num2str(i));
    end
    if(ObsAnom(i) == 1)
        %plot(Dfx, Dfy, 'r');
        plot(Dfx(1), Dfy(1), 'g.');
        %text(Dfx(5), Dfy(5), num2str(i));
        %plot(Dfx(1), Dfy(1), 'wsquare');
    end
    hold on;
end
set(h4, 'YDir', 'reverse');
toc;

%{


%%
for i=1:numTrajec
    if(Votev
    ObsAnomCount = sum(VoteVec(i) > 2);
%%
for i = 1:numTrajec
    if(VoteVec(i) == 1 & AnomVec(i) == 1)
        tp(i) = 1;
    end
end
tpcount = sum(tp);

for i = 1:numTrajec
    if(VoteVec(i) == 0 & AnomVec(i) == 1)
        tn(i) = 1;
    end
end
tncount = sum(tn);

for i = 1:numTrajec
    if(VoteVec(i) == 1 & AnomVec(i) == 0)
        fp(i) = 1;
    end
end
fpcount = sum(fp);

for i = 1:numTrajec
    if(VoteVec(i) == 0 & AnomVec(i) == 0)
        fn(i) = 1;
    end
end
fncount = sum(fn);

P = tp/(tp+fp);
R = tp/(tp+fn);

disp(R);
disp(P);
%}
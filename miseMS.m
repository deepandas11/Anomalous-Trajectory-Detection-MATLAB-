%function [clustCent,data2cluster,cluster2dataCell] = miseMS(dataPts);
clc;
feat = shape(:,4);
mx = max(feat);
mn = min(feat>0);
dim = mx-mn;
initbw = mn;
lim = mx;
skip = 0.1*dim;
aq = [];
for i = initbw:skip:(0.7*lim)
    aq = [aq i];
end

lenaq = size(aq,2);
errorsum = zeros(1, lenaq);
%%
for i = 1:lenaq
    bw = aq(i);
    [clustcent, data2clus, clustMemCell] = HGMeanShiftCluster(feat', bw, 'gaussian');
    sum = 0;
    for j = 1:numTrajec
        p = data2clus(j);
        tf = feat(j,:);     
        tf1 = tf';
        cc= clustcent(:,p);
        d1 = dist(tf,cc);
        sum = sum+d1;    
    end
    errorsum(i) = sum; 
end




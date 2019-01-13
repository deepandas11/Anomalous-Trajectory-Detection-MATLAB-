function [clustCent,data2cluster,cluster2dataCell] = miseMS(dataPts);
feat = dataPts;
mx = max(feat);
mn = min(feat);
dim = mx-mn;
initbw = min(dim);
lim = max(dim);

aq = [];
for i = initbw:initbw:(0.7*lim)
    aq = [aq i];
end

lenaq = length(aq);
errorsum = zeros(1, lenaq);

for i = 1:lenaq
    bw = aq(i);
    [clustcent, data2clus, clustMemCell ] = HGMeanShift(feat, bw, 'gaussian');
    sum = 0;
    for j = 1:numTrajec
        p = point2cluster(i);
        sum = sum+ dist(feat(i,:),clustCent(p,:));
    end
    errorsum(i) = sum;
end
end

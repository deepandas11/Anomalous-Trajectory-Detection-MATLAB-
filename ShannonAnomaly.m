function [H,thresh,lambda] = ShannonAnomaly(nfeat, nclusters,clustCent,l);

% -----------------------------------------------------------------------
% -- INPUT --
% trajec_feat = The feature space of all trajectories
% nclusters = number of clusters determined from mean shift clustering
% -----------------------------------------------------------------------
% --OUTPUT--
% H = The Shannon Entropy Abnormality score for each trajectory based on
% it's feature vector 
% -----------------------------------------------------------------------


lambda = l;
thresh = lambda*(log(nclusters));
ntrac = size(nfeat,1);
H = zeros(ntrac,1);

for i = 1: ntrac
    tf = nfeat(i,:);
    tf1 = tf';
    d = zeros(1,nclusters);
    dsum = 0;
    for j=1:nclusters
        cc = clustCent(:,j);
        d(j)= dist(tf,cc);
        dsum = dsum+d(j);
    end
    psh = zeros(1,nclusters);
    for k=1:nclusters
        psh(k) = d(k)/dsum;
    end
    for l = 1:nclusters
        H(i) = H(i)+ psh(l)*(log(psh(l)/log(2)));
    end
    H(i) = -1*H(i);
end






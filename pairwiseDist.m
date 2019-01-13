function distance = pairwiseDist(prefSet1, prefSet2)

%--------------------------------------------------------------------------
% This function calculates the distance between the preference sets of two
% trajectories, which is defined in eq(6) in he paper[1].
%--------------------------------------------------------------------------
intersection = prefSet1.*prefSet2;
union = prefSet1 + prefSet2;
union =union - intersection;
union = sum(union);
intersection = sum(intersection);
distance = (union - intersection)/union;
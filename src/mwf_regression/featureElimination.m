function [Xtr_f1, Xtr_f2] = featureElimination(Xtr, ytr)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
C = corr(Xtr, ytr, 'Rows','complete');   % returns [nFeatures x 1] correlations
absC = abs(C);                           % absolute values (strength only)

thresh = 0.02;  % adjust based on histogram
keep = absC > thresh;

Xtr_f1  = Xtr(:, keep);

v = var(Xtr, 0, 1);
th = 1e-6;                        % start here; tighten/loosen as needed
keep = v > th;

Xtr_f2 = Xtr(:, keep);
end
function [Xtr_dmg, ytr_dmg, Xval_dmg, yval_dmg] = filterDamageOnly(net, Xtr, ytr, Xval, yval)
%FILTERDAMAGEONLY Keep only samples predicted as Damage (label = 1)
%   net      : trained network
%   Xtr, ytr : training data & labels (0 = Noise, 1 = Damage)
%   Xval,yval: validation data & labels
%
%   Returns subsets (Xtr_dmg, ytr_dmg, Xval_dmg, yval_dmg)


[mu, sigma] = deal(mean(Xtr,1), std(Xtr,0,1));
sigmaFloor = 1e-3;             % adjust threshold so values are not blowing up
sigmaClamped = max(sigma, sigmaFloor);

Xtr_p  = (Xtr  - mu) ./ sigmaClamped;
Xval_p = (Xval - mu) ./ sigmaClamped;

    % --- Predict on training data ---
    scores = predict(net, Xtr_p);    % size: numSamples × 2
    [~, idx] = max(scores,[],2);   % find max along classes for each sample
    keepIdx = (idx == 2) ;             % logical vector, true where prediction is class "1"
    Xtr_dmg = double(Xtr(keepIdx,:));    % keep only those inputs (if X is tabular/array)
    ytr_dmg = ytr(keepIdx);    % keep corresponding labels for training data

    scores_val = predict(net, Xval_p);  % predict on validation data
    [~, idx_val] = max(scores_val, [], 2);  % find max along classes for validation
    keepIdx_val = (idx_val == 2);  % logical vector for validation data
    Xval_dmg = double(Xval(keepIdx_val, :));  % keep only validation inputs predicted as damage
    yval_dmg = yval(keepIdx_val);  % keep corresponding labels for validation data

 
end

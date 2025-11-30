function [X_pca, score, coeff, explained, nPCs] = runPCAforTraining(X, varThreshold)
%RUNPCAFORTRAINING Standardize data, run PCA, and select PCs
%   [X_pca, score, coeff, explained, nPCs] = runPCAforTraining(X, varThreshold)
%
% Inputs:
%   X            - Feature matrix (N × D)
%   varThreshold - Variance threshold in % (default = 95)
%
% Outputs:
%   X_pca     - Reduced data (N × nPCs) ready for training
%   score     - PCA-transformed full dataset (N × D)
%   coeff     - PCA loadings (D × D)
%   explained - Variance explained by each PC (%)
%   nPCs      - Number of PCs chosen to reach varThreshold

    if nargin < 2
        varThreshold = 95;  % default: keep 95% variance
    end

    % --- Step 1: Standardize features (zero mean, unit variance) ---
    Xstd = zscore(X);

    % --- Step 2: Run PCA ---
    [coeff, score, ~, ~, explained] = pca(Xstd);

    % --- Step 3: Decide how many PCs to keep ---
    cumVar = cumsum(explained);
    nPCs = find(cumVar >= varThreshold, 1, 'first');

    % --- Step 4: Return reduced dataset ---
    X_pca = score(:, 1:nPCs);

    % --- Reporting ---
    fprintf('PCA complete: %d features reduced to %d PCs (%.2f%% variance retained).\n', ...
        size(X,2), nPCs, cumVar(nPCs));

    % Optional: show variance explained plot
    figure;
    pareto(explained);
    xlabel('Principal Component'); ylabel('Variance Explained (%)');
    title(sprintf('PCA Variance Explained (retaining %d PCs = %.2f%%)', nPCs, cumVar(nPCs)));
end

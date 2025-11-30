function [acc, ypred_true] = computeValidationAccuracy(net, Xval, yval)
% COMPUTEVALIDATIONACCURACY  Calculate validation accuracy for a trained dlnetwork.
%   net   : trained dlnetwork (MLP or CNN)
%   Xval  : validation features, rows = samples, columns = features
%   yval  : validation labels (categorical or numeric)
%
% Returns:
%   acc   : validation accuracy in percentage (0–100)

    % Ensure labels are categorical
    if ~iscategorical(yval)
        yval = categorical(yval);
    end

    % Create datastore
    dsX = arrayDatastore(Xval, 'IterationDimension', 1);
    mb = 128;
    % Predict scores/logits with minibatchpredict
    Yscores = minibatchpredict(net, dsX, MiniBatchSize=mb, InputDataFormats="BC"); % [N x K]

    % Convert to predicted class
    if size(Yscores, 2) == 1
        % Sigmoid output → threshold at 0.5
        ypred = categorical(Yscores >= 0.5);
    else
        % Softmax output → choose highest probability column
        [~, idx] = max(Yscores, [], 2);
        cats = categories(yval);  % maintain original label order
        if numel(cats) == size(Yscores, 2)
            ypred = categorical(cats(idx));
        else
            % fallback: numeric categorical (1..K)
            ypred = categorical(idx);
        end
    end

    % Compute accuracy
    acc = mean(ypred == yval) * 100;
    ypred_true = ypred == yval;
    fprintf('Validation Accuracy: %.2f %% (%d/%d correct)\n', ...
        acc, nnz(ypred == yval), numel(yval));
end

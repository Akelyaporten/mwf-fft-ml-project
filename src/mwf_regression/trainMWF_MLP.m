function [net, mu, sigma, metrics] = trainMWF_MLP(Xtr, ytr, Xval, yval, Xte, yte)
    
% --- Standardize features using TRAINING stats only ---
    [Xtr, mu, sigma] = zscore(Xtr);                     % mu, sigma from train
    Xval = (Xval - mu) ./ sigma;
    Xte  = (Xte  - mu) ./ sigma;

 layers = [
    featureInputLayer(22507, "Normalization","none")

    fullyConnectedLayer(1024)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.4)

    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(32)
    reluLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(1)
];

   % --- Layers ---

%    layers = [
%     featureInputLayer(22507, "Normalization","none")
% 
%     fullyConnectedLayer(4096)
%     reluLayer
%     dropoutLayer(0.6)
% 
%     fullyConnectedLayer(2048)
%     reluLayer
%     dropoutLayer(0.4)
% 
%     fullyConnectedLayer(1024)
%     reluLayer
%     dropoutLayer(0.3)
% 
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.3)
% 
%     fullyConnectedLayer(256)
%     reluLayer
%     dropoutLayer(0.3)
% 
%     fullyConnectedLayer(128)
%     reluLayer
%     dropoutLayer(0.3)
% 
%     fullyConnectedLayer(64)
%     reluLayer
%     dropoutLayer(0.2)
% 
%     fullyConnectedLayer(64)
%     reluLayer
%     dropoutLayer(0.2)
% 
%     fullyConnectedLayer(32)
%     reluLayer
%     dropoutLayer(0.1)
% 
%     fullyConnectedLayer(32)
%     reluLayer
%     dropoutLayer(0.1)
% 
%     fullyConnectedLayer(1)
% ];

% Xtr= reshape(permute((Xtr), [2 3 4 1]), [22507 1 1 size(Xtr,1)]);
% Xval= reshape(permute((Xval), [2 3 4 1]), [22507 1 1 size(Xval,1)]);
% Xte= reshape(permute((Xte), [2 3 4 1]), [22507 1 1 size(Xte,1)]);
% 
% ytr  = ytr(:)';   % [1 × N]
% yval = yval(:)';
% yte  = yte(:)';
% 
% Validation datastore
dsXval = arrayDatastore(Xval,'IterationDimension',1);
dsYval = arrayDatastore(yval,'IterationDimension',1);
dsVal = combine(dsXval,dsYval);
% 
% layers = [
%    imageInputLayer([22507 1 1], "Normalization","none")
% 
%    convolution2dLayer([7 1],32,"Padding","same"); reluLayer
%    maxPooling2dLayer([4 1],"Stride",[4 1])
% 
%    convolution2dLayer([7 1],32,"Padding","same"); reluLayer
%    maxPooling2dLayer([4 1],"Stride",[2 1])
% 
%    convolution2dLayer([7 1],64,"Padding","same"); reluLayer
%    maxPooling2dLayer([2 1],"Stride",[2 1])
% 
%    convolution2dLayer([7 1],128,"Padding","same"); reluLayer
%    maxPooling2dLayer([2 1],"Stride",[2 1])
% 
%    globalAveragePooling2dLayer
% 
%    fullyConnectedLayer(256); reluLayer; dropoutLayer(0.2)
%    fullyConnectedLayer(64);  reluLayer
%    fullyConnectedLayer(32); reluLayer
%    fullyConnectedLayer(1)
%    ];
% 

   
    % --- Training options (Adam, L2, etc.) ---
    mb = 64;
    % itersPerEpoch = ceil(size(Xtr,1)/mb);
    options = trainingOptions("adam", ...
        "InitialLearnRate", 1e-4, ...
        "SquaredGradientDecayFactor", 0.999, ...
        "GradientDecayFactor", 0.9, ...
        "MaxEpochs", 10, ...
        "MiniBatchSize", mb, ...
        "Shuffle", "every-epoch", ...
        "L2Regularization", 1e-4, ...
        "ValidationData", dsVal, ...
        "ValidationFrequency", 5, ... %max(1, floor(0.25*itersPerEpoch)), ...
        "Plots","training-progress", ...
        "Verbose", true);

    net = dlnetwork(layers);

    % Training datastore
    
    dsX = arrayDatastore(Xtr,'IterationDimension',1);
    dsY = arrayDatastore(ytr,'IterationDimension',1);
    dsTrain = combine(dsX,dsY);
    
    dsXte  = arrayDatastore(Xte,  "IterationDimension", 1);
    dsYte  = arrayDatastore(yte,  "IterationDimension", 1);
    dsTest = combine(dsXte, dsYte);
    
    
    % Then:
    net = trainnet(dsTrain, net, "huber", options);


    % --- Evaluate ---
    yhat_val = minibatchpredict(net, dsVal);
    yhat_te  = minibatchpredict(net, dsTest);


    metrics.val   = computeRegMetrics(yval, yhat_val);
    metrics.test  = computeRegMetrics(yte,  yhat_te);
    
  
    fprintf('Validation: RMSE=%.5g  MAE=%.5g  R^2=%.4f\n', ...
        metrics.val.RMSE, metrics.val.MAE, metrics.val.R2);
    fprintf('Test      : RMSE=%.5g  MAE=%.5g  R^2=%.4f\n', ...
        metrics.test.RMSE, metrics.test.MAE, metrics.test.R2);
    
    % --- Helper function ---
    function m = computeRegMetrics(ytrue, ypred)
        ytrue = (ytrue(:)); 
        ypred = (ypred(:));
        m.RMSE = sqrt(mean((ypred - ytrue).^2));
        m.MAE  = mean(abs(ypred - ytrue));
        ssRes  = sum((ypred - ytrue).^2);
        ssTot  = sum((ytrue - mean(ytrue)).^2);
        m.R2   = 1 - ssRes / ssTot;
    end
end
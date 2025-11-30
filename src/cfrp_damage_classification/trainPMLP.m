function [net] = trainPMLP(Xtr, ytr, Xval, yval)


[mu, sigma] = deal(mean(Xtr,1), std(Xtr,0,1));
sigmaFloor = 1e-3;             % adjust threshold so values are not blowing up
sigmaClamped = max(sigma, sigmaFloor);

Xtr  = (Xtr  - mu) ./ sigmaClamped;
Xval = (Xval - mu) ./ sigmaClamped;

cats = categories(categorical(ytr));
ytr  = categorical(ytr, cats);
yval = categorical(yval, cats);


dsXval = arrayDatastore(Xval,'IterationDimension',1);
dsYval = arrayDatastore(yval,'IterationDimension',1);
dsVal = combine(dsXval,dsYval);


dsX = arrayDatastore(Xtr,'IterationDimension',1);
dsY = arrayDatastore(ytr,'IterationDimension',1);
dsTrain = combine(dsX,dsY);



% LAYERS (1D conv)
 layers = [
    featureInputLayer(126, "Normalization","none")

    fullyConnectedLayer(1024)
    reluLayer
    dropoutLayer(0.4)

    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)

    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(32)
    reluLayer
 

    fullyConnectedLayer(4)
    softmaxLayer
];


% Training options for trainnet
options = trainingOptions("adam", ...
    "InitialLearnRate",1e-5, ...
    "MaxEpochs",900, ...
    "MiniBatchSize",64, ...
    "Shuffle","every-epoch", ...
    "L2Regularization",1e-4, ...
    "ValidationData",dsVal, ...
    "ValidationFrequency",10, ...
    "Verbose",true, ...    
    "Metrics",["accuracy"], ...                  % report accuracy
    "Plots","training-progress");                % live plot window

% Train with cross-entropy loss
net = trainnet(dsTrain, layers, "crossentropy", options);
end
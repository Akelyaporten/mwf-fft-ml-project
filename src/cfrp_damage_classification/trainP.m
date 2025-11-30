function [net] = trainP(Xtr, ytr, Xval, yval)


[mu, sigma] = deal(mean(Xtr,1), std(Xtr,0,1));
sigmaFloor = 1e-3;             % adjust threshold so values are not blowing up
sigmaClamped = max(sigma, sigmaFloor);

Xtr  = (Xtr  - mu) ./ sigmaClamped;
Xval = (Xval - mu) ./ sigmaClamped;

cats = categories(categorical(ytr));
ytr  = categorical(ytr, cats);
yval = categorical(yval, cats);

D   = size(Xtr,2);
Xtr  = reshape(permute(Xtr,[2 3 4 1]), [D 1 1 size(Xtr,1)]);
Xval = reshape(permute(Xval,[2 3 4 1]), [D 1 1 size(Xval,1)]);

dsXval = arrayDatastore(Xval,'IterationDimension',4);
dsYval = arrayDatastore(yval,'IterationDimension',1);
dsVal = combine(dsXval,dsYval);


dsX = arrayDatastore(Xtr,'IterationDimension',4);
dsY = arrayDatastore(ytr,'IterationDimension',1);
dsTrain = combine(dsX,dsY);



% LAYERS (1D conv)
layers = [
    imageInputLayer([D 1 1], "Normalization","none","Name","in")

    convolution2dLayer([7, 1], 64, "Padding","same"); 
    batchNormalizationLayer; 
    reluLayer
    

    convolution2dLayer([7, 1], 64, "Padding","same"); 
    batchNormalizationLayer; 
    reluLayer

    maxPooling2dLayer([2 1],"Stride",[2 1])

    dropoutLayer(0.2)
    

    convolution2dLayer([7, 1],128, "Padding","same"); 
    batchNormalizationLayer; 
    reluLayer

    convolution2dLayer([7, 1],128, "Padding","same"); 
    batchNormalizationLayer; 
    reluLayer
    
    maxPooling2dLayer([2 1],"Stride",[2 1])

    dropoutLayer(0.3)

    convolution2dLayer([7, 1],256, "Padding","same"); 
    batchNormalizationLayer; 
    reluLayer

    globalAveragePooling2dLayer
    
    dropoutLayer(0.4)
       
    fullyConnectedLayer(5)
    softmaxLayer
];

% Training options for trainnet
options = trainingOptions("adam", ...
    "InitialLearnRate",1e-5, ...
    "MaxEpochs",20, ...
    "MiniBatchSize",64, ...
    "Shuffle","every-epoch", ...
    "L2Regularization",1e-3, ...
    "ValidationData",dsVal, ...
    "ValidationFrequency",10, ...
    "ValidationPatience",10, ...
    "Verbose",true, ...    
    "Metrics",["accuracy"], ...                  % report accuracy
    "Plots","training-progress");                % live plot window

% Train with cross-entropy loss
net = trainnet(dsTrain, layers, "crossentropy", options);


end
function [yhat_val, yhat_te] = randomForest(Xtr, ytr)
t = templateTree('MaxNumSplits',12, 'MinLeafSize',20);
mdl_bag = fitrensemble(Xtr, ytr, ...
   'Method','Bag', 'Learners',t, 'NumLearningCycles',200, ...
   'Options', statset('UseParallel',true));

yhat_val = predict(mdl_bag, Xval);
yhat_te  = predict(mdl_bag, Xte);

end
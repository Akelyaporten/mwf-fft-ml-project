function [Ytrain, Yval] = binaryLabels(Label_train,Label_test)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
isNoise = Label_train == "Noise";         % logical vector
Ytrain = categorical(double(~isNoise));  
% '0' = noise → class 'Noise'
% '1' = others → class 'Defect'

isNoise_t = Label_test == "Noise";         % logical vector
Yval = categorical(double(~isNoise_t));  

% '0' = noise → class 'Noise'
% '1' = others → class 'Defect's

end
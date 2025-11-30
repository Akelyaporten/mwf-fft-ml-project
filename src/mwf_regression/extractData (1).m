%% ------------------------------------------------------
% Feature and Response Extraction for MWF Data
% Author: Ali Çelikkol
% Date: 2025-08-24
%
% Description:
%   Loads a .mat file containing the MWF dataset.
%   Extracts:
%       - Features: Columns 1–22507 + column 22509 (Section info)
%       - Response: Column 22512 (distance sensor ↔ drilling location)
%
% Output:
%   X : Training data matrix
%   y : Response variable vector
%
% ------------------------------------------------------
function [X, y] = extractData(filePath)
feature = load(filePath);

fn = fieldnames(feature);

feature = feature.(fn{1});

%% Extract features (X) and response (y)
% Features: columns 1:22511 and 22509

X = double(feature{:, 1:22511});

% Response: column 22512
y = double(feature{:, 22512});

%% Display dataset info
fprintf('Feature matrix X size: %d rows x %d columns\n', size(X,1), size(X,2));
fprintf('Response vector y size: %d rows x 1 column\n', size(y,1));

% --- Save next to the input file ---
[folder, base, ~] = fileparts(filePath);
outFile = fullfile(folder, [base '_FeatureResponse.mat']);
save(outFile(2), 'X', 'y');
fprintf('Extracted X and y saved to %s\n', outFile);
end
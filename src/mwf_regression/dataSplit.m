function [Xtr, ytr, Xval, yval, Xte, yte] = dataSplit(X, y, valFolder, testFolder)

folderCol = 5432; % column index for Folder % change it to 22510 later again if it is not so.

% --- Extract all rows where Folder == targetFolder ---
rowsVal = X(:, folderCol) == valFolder;
rowsTest = X(:, folderCol) == testFolder; 

% change column form : to 22507)
Xval = X(rowsVal, 1:5431);
yval = y(rowsVal);
Xte = X(rowsTest, 1:5431);
yte = y(rowsTest);
% Extract training data (not in validation or test folders)
Xtr = X(~(rowsVal | rowsTest), 1:5431);
ytr = y(~(rowsVal | rowsTest));

 % --- Save next to the input file ---
[folder, base, ~] = fileparts("/MATLAB Drive/Project1/DataSplit.m");
outFile = fullfile(folder, [base 'splittedDataset_1_3.mat']);
save(outFile(2), 'Xtr', 'ytr', "Xval", "yval", "Xte", "yte");
fprintf('Extracted X and y saved to %s\n', outFile);

end
% Task E
% Prepare train-test data for taskE3

% Load data
data = load('cw1e.mat');
x = data.x;
y = data.y;
xy = [x, y];

% Random train-test splits
rng(1)
a = 0.8; % Fraction of data for training
M = int16(a*length(x));
random_xy = xy(randperm(size(xy, 1)), :); % Randomly switch rows
xy_train = random_xy(1:M,:); % Select train set
save("xy_train.mat", "xy_train") % Save train set to file

xy_test = random_xy(M+1:length(x), :); % Select test set
save("xy_test.mat", "xy_test") % Save test set to file
data = load('cw1e.mat');
x = data.x;
y = data.y;
xy = [x, y];
a = 0.8;
N = int16(a*length(x));
rng(1)
random_xy = xy(randperm(size(xy, 1)), :);
xy_train = random_xy(1:N,:);
x_train = xy_train(:, 1:2);
y_train = xy_train(:, 3);
xy_test = random_xy(N+1:length(x), :);

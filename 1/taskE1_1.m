% Task E
% 2D (input) data
% Model with single Squared Exponential covariance function with Automatic
% Relevance Detemination (ARD) distance measure; covSEard.
% Produce 3D predictive plot

% Load data
data = load('cw1e.mat');
x = data.x;
y = data.y;
a = 8; % Range for input test data
N = 70; % Number of input test data samples

% Display data
%mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));

mean_func = []; % empty - don't use mean function
cov_func = @covSEard; % squared exponential covariance function
lik_func = @likGauss; % gaussian likelihood func

%initial hyperparams
cov = 0.1*randn(3,1); % initial covariance: 1) log length-scale1, 2) log length-scale2, 3) log signal std-dev
lik = 0; % initial likelihood - log noise st dev
hyp = struct('mean', [], 'cov', cov, 'lik', lik); % hyperparameter struct

% optimised hyperparams by minimising negative log likelihood
hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, mean_func, cov_func, lik_func, x, y);

disp(hyp_opt)
disp(hyp_opt.cov)

[xs1, xs2] = meshgrid(linspace(-a, a, N)', linspace(-a, a, N)');
xs = [xs1(:), xs2(:)];
[mu, s2] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y, xs);

% Produce 3D plot of predictive mean for test input, with training data overlayed.
plot1 = scatter3(x(:,1), x(:,2), y, 'r', 'filled', 'DisplayName', "Training Data");
hold on;
plot2 = mesh(xs1, xs2, reshape(mu, N, N), 'DisplayName', "Predictive Mean");
colormap(cool);
xlabel('x1')
ylabel('x2')
zlabel('y')
xlim([-a, a])
ylim([-a, a])
lgnd = legend('show');
lgnd.Position = [0.22 0.75 0.1 0.1];
%view(150, 60)
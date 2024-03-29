% Task E
% 2D (input) data
% Model with single Squared Exponential covariance function with Automatic
% Relevance Detemination (ARD) distance measure; covSEard.
% Produce 2D plots with 95% confidence bands, varying FIRST input

rng(1)
% Load data
data = load('cw1e.mat');
x = data.x;
y = data.y;
a = 8; % Range for input test data
N = 100; % Number of input test data samples
input2 = 5;
xs = [linspace(-a, a, N)', zeros(N, 1)+input2]; % Test data, varying first input, second held constant

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

% predictions
[mu, s2] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y, xs);

% 2D plots of predictive mean against one input, with 95% confidence bands
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
fill([xs(:,1); flip(xs(:,1),1)], f, [7 7 7]/8, 'DisplayName', "95% Error Bounds")
hold on;
plot(xs(:,1), mu, 'r', 'DisplayName', "Predictive Mean");
xlabel('x1');
ylabel('y')
lgnd = legend('show');
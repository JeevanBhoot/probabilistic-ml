% Task E
% Find best model for both covariance function settings:
% 1) Single Squared Exponential func,
% 2) Sum of two Squared Exponential funcs,
% out of N different random initial settings of the hyperparams.
% Return marginal likelihood of each to compare both models.

% Load data
data = load('cw1e.mat');
x = data.x[];
y = data.y;

N = 250; % Number of randomly initialised models to optimize

mean_func = []; % empty - don't use mean function
cov_func1 = @covSEard; % squared exponential covariance function
cov_func2 = {@covSum, {@covSEard, @covSEard}}; % sum of two squared exponential cov funcs
lik_func = @likGauss; % gaussian likelihood func

model1_nlls = zeros(N,1); % All negative log marginal likelihoods for model 1
model2_nlls = zeros(N,1); % All negative log marginal likelihoods for model 2

for i = 1:N
    cov1 = 0.1*randn(3,1);
    cov2 = 0.1*randn(6,1);
    hyp1 = struct('mean', [], 'cov', cov1, 'lik', 0);
    hyp2 = struct('mean', [], 'cov', cov2, 'lik', 0);
    [nlZ, ~] = gp(hyp1, @infGaussLik, mean_func, cov_func1, lik_func, x, y);
    model1_nlls(i) = nlZ;
    [nlZ, ~] = gp(hyp2, @infGaussLik, mean_func, cov_func2, lik_func, x, y);
    model2_nlls(i) = nlZ;
end

model1_best = min(model1_nlls);
model2_best = min(model2_nlls);


disp(model1_best)
disp(exp(-model1_best))

disp(model2_best)
disp(exp(-model2_best))

%disp("MODEL 1: Best Negative Log Marginal Likelihood = ", model1_best)
    %"Best Marginal Likelihood = ", exp(-model1_best), "MODEL 2:", ...
    %"Best Negative Log Marginal Likelihood = ", model2_best, ...
    %"Best Marginal Likelihood = ", exp(-model2_best))
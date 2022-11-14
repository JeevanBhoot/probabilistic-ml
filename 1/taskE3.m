% Task E
% Find best model for both covariance function settings:
% 1) Single Squared Exponential func,
% 2) Sum of two Squared Exponential funcs,
% out of N different random initial settings of the hyperparams.
% Return marginal likelihood of each to compare both models.

%rng(1)
% Load data
xy_train = load("xy_train.mat").xy_train;
xy_test = load("xy_test.mat").xy_test;

x_train = xy_train(:, 1:2); % Split train into x and y
y_train = xy_train(:, 3);

x_test = xy_test(:, 1:2); % Split test into x and y
y_test = xy_test(:, 3);

N = 50; % Number of randomly initialised models to optimize

mean_func = []; % empty - don't use mean function
cov_func1 = @covSEard; % squared exponential covariance function
cov_func2 = {@covSum, {@covSEard, @covSEard}}; % sum of two squared exponential cov funcs
lik_func = @likGauss; % gaussian likelihood func

model1_nlls = zeros(N,1); % All negative log marginal likelihoods for model 1
model2_nlls = zeros(N,1); % All negative log marginal likelihoods for model 2

for i = 1:N
    cov1 = 1*randn(3,1);
    cov2 = 1*randn(6,1);
    hyp1 = struct('mean', [], 'cov', cov1, 'lik', 0);
    hyp2 = struct('mean', [], 'cov', cov2, 'lik', 0);
    
    hyp1_opt = minimize(hyp1, @gp, -100, @infGaussLik, mean_func, cov_func1, lik_func, x_train, y_train);
    [nlZ, ~] = gp(hyp1_opt, @infGaussLik, mean_func, cov_func1, lik_func, x_train, y_train);
    model1_nlls(i) = nlZ;
    model1s(i) = hyp1_opt;
    
    hyp2_opt = minimize(hyp2, @gp, -100, @infGaussLik, mean_func, cov_func2, lik_func, x_train, y_train);
    [nlZ, ~] = gp(hyp2_opt, @infGaussLik, mean_func, cov_func2, lik_func, x_train, y_train);
    model2_nlls(i) = nlZ;
    model2s(i) = hyp2_opt;
end

[model1_best_nll, i1] = min(model1_nlls);
[model2_best_nll, i2] = min(model2_nlls);

hyp1_opt = model1s(i1);
hyp2_opt = model2s(i2);

[mu_1, s2_1] = gp(hyp1_opt, @infGaussLik, mean_func, cov_func1, lik_func, x_train, y_train, x_test);
[mu_2, s2_2] = gp(hyp2_opt, @infGaussLik, mean_func, cov_func2, lik_func, x_train, y_train, x_test);

error1 = mu_1 - y_test;
error1 = error1.^2;
error2 = mu_2 - y_test;
error2 = error2.^2;

disp("Model One:")
disp(hyp1_opt)
disp(hyp1_opt.cov)
disp(model1_best_nll)
disp(sqrt(sum(error1)/length(y_test)))
disp("Model Two:")
disp(hyp2_opt)
disp(hyp2_opt.cov)
disp(model2_best_nll)
disp(sqrt(sum(error2)/length(y_test)))
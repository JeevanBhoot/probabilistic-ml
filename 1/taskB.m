data = load('cw1a.mat'); % load data
x = data.x;
y = data.y;

mean_func = []; % empty - don't use mean function
cov_func = @covSEiso; % squared exponential covariance function
lik_func = @likGauss; % gaussian likelihood func

% different initial hyperparams than taskAdata = load('cw1a.mat'); % load data
x = data.x;
y = data.y;

mean_func = []; % empty - don't use mean function
cov_func = @covSEiso; % squared exponential covariance function
lik_func = @likGauss; % gaussian likelihood func

%different initial hyperparams than taskA
cov = [-2, 2]; % initial covariance
lik = -1; % initial likelihood
hyp = struct('mean', [], 'cov', cov, 'lik', lik); % hyperparameter struct

% optimised hyperparams by minimising negative log likelihood
hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, mean_func, cov_func, lik_func, x, y);

disp(hyp_opt) %| cov = [-2.0540 -0.1087] | lik = -2.1385


xs = linspace(-3.5, 3.5, 500)'; % test data in range -3.5 to 3.5
                               % min(x) = -2.8966, max(x) = 2.5093
% make predictions
[mu, s2] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y, xs);
% mu - pred mean, s2 - pred std dev

% plot predictive mean at test points with 95% confidence bounds
% as well as training data overlayed
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on;
plot(xs, mu);
plot(x, y, 'b+')
xlabel('x');
ylabel('y')
legend('95% Error Bounds', 'Predictive Mean', 'Training Datapoints');
cov = [-1, 0]; % initial covariance
lik = 0; % initial likelihood

% initialise hyperparameter struct
hyp = struct('mean', [], 'cov', cov, 'lik', lik);
% optimised hyperparams by minimising negative log likelihood
hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, mean_func, cov_func, lik_func, x, y);

%disp(hyp_opt) | cov = [-2.0540 -0.1087] | lik = -2.1385


xs = linspace(-3.5, 3.5, 500)'; % test data in range -3.5 to 3.5
                               % min(x) = -2.8966, max(x) = 2.5093
% make predictions
[mu, s2] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y, xs);
% mu - pred mean, s2 - pred std dev

% plot predictive mean at test points with 95% confidence bounds
% as well as training data overlayed
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on;
plot(xs, mu);
plot(x, y, 'b+')
xlabel('x');
ylabel('y')
legend('95% Error Bounds', 'Predictive Mean', 'Training Datapoints');
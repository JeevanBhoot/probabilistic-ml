% Task A
% Train a GP with a squared exponential covariance function
% Start the log hyper-parameters at hyp.cov = [-1 0]; hyp.lik = 0
% Produce predictive plot with 95% error bounds

data = load('cw1a.mat'); % load data
x = data.x;
y = data.y;

mean_func = []; % empty - don't use mean function
cov_func = @covSEiso; % squared exponential covariance function
lik_func = @likGauss; % gaussian likelihood func

%initial hyperparams
cov = [-1, -10]; % initial covariance: 1) log length-scale, 2) log signal std-dev
lik = 0; % initial likelihood - log noise st dev
hyp = struct('mean', [], 'cov', cov, 'lik', lik); % hyperparameter struct
[nlZ, ~] = gp(hyp, @infGaussLik, mean_func, cov_func, lik_func, x, y); % initial NLML
% optimised hyperparams by minimising negative log likelihood
hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, mean_func, cov_func, lik_func, x, y);
[nlZ2, ~] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y); % optimised NLML
disp(nlZ)
disp(nlZ2)
disp(hyp_opt)


xs = linspace(-4, 4, 750)'; % test data in range -3.5 to 3.5
                               % min(x) = -2.8966, max(x) = 2.5093
% make predictions
[mu, s2] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y, xs);
% mu - pred mean, s2 - pred std dev

% plot predictive mean at test points with 95% confidence bounds
% as well as training data overlayed
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on;
plot(xs, mu, 'r');
plot(x, y, 'b+')
xlabel('x');
ylabel('y')
legend('95% Error Bounds', 'Predictive Mean', 'Training Datapoints');
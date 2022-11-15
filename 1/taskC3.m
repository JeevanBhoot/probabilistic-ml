data = load('cw1a.mat');
x = data.x;
y = data.y;

mean_func = []; % empty - don't use mean function
cov_func = @covPeriodic; % periodic covariance function
lik_func = @likGauss; % gaussian likelihood func

%initial hyperparams
cov = [0, 0, 0]; % initial covariance: 1) log length-scale, 2) log period, 3) log signal st-dev
lik = 0; % initial likelihood - log noise st dev
hyp = struct('mean', [], 'cov', cov, 'lik', lik); % hyperparameter struct

% optimised hyperparams by minimising negative log likelihood
[nlZ, ~] = gp(hyp, @infGaussLik, mean_func, cov_func, lik_func, x, y);
hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, mean_func, cov_func, lik_func, x, y);
[nlZ2, ~] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y);
disp(nlZ)
disp(nlZ2)
disp(hyp_opt)

xs = linspace(-4, 4, 750)'; % test data in range -3.5 to 3.5
                               % min(x) = -2.8966, max(x) = 2.5093
% make predictions
[mu, s2] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y, x);
% mu - pred mean, s2 - pred std dev

% plot predictive mean at test points with 95% confidence bounds
% as well as training data overlayed
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
%fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on;
%plot(x, mu, 'r+');
%plot(x, y, 'b+')
%xlabel('x');
%ylabel('y')
%legend('95% Error Bounds', 'Predictive Mean', 'Training Datapoints');
histogram(y-mu)
xlabel('Residual Error')
% Task D
% Generate random (essentially) noise free functions evaluated at x =
% linspace(-5,5,200)'
% Covariance function: {@covProd, {@covPeriodic, @covSEiso}}, 
% with covariance hyperparameters hyp.cov = [-0.5 0 0 2 0]

mean_func = []; % empty - don't use mean function
cov_func = {@covProd, {@covPeriodic, @covSEiso}}; % squared exponential covariance function
lik_func = @likGauss; % gaussian likelihood func

%initial hyperparams
cov = [-0.5, 0, 0, 2, 0]; % initial covariance params
lik = 0; % initial likelihood - log noise st dev

x = randn(200, 1);
xs = linspace(-5, 5, 200)';
K = feval(cov_func{:}, cov, xs) + 1e-6*eye(200);
K_per = feval(@covPeriodic, [-0.5, 0, 0], xs) + 1e-6*eye(200);
K_iso = feval(@covSEiso, [2, 0], xs) + 1e-6*eye(200);

y = chol(K)' * x;
y_per = chol(K_per)' * x;
y_iso = chol(K_iso)' * x;

hold on;
plot(xs, y, 'DisplayName', "Product")
plot(xs, y_per, 'r', 'DisplayName', "Periodic")
plot(xs, y_iso, 'g', 'DisplayName', "Squared Exponential")
Lgnd = legend('show');





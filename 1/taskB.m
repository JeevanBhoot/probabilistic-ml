% Task B
% Show that by initializing the hyperparameters differently, 
% you can find a different local optimum for the hyperparameters.
% Hyperparameter search, producing 2D contour plot of negative log marginal
% likelihood as a function of two hyperparameters.

data = load('cw1a.mat'); % load data
x = data.x;
y = data.y;

mean_func = []; % empty - don't use mean function
cov_func = @covSEiso; % squared exponential covariance function
lik_func = @likGauss; % gaussian likelihood func

%initial hyperparams
N = 20; % number of hyperparams to test
cov_vals = linspace(-5, 5, N); % range of values for covariance hyperparams
lik = 0; % initial likelihood - log noise st dev
grid = zeros(N, N);

for i = 1:N
    for j = 1:N
        hyp = struct('mean', [], 'cov', [cov_vals(i), 0], 'lik', cov_vals(j)); % hyperparameter struct
        hyp_opt = minimize(hyp, @gp, -100, @infGaussLik, mean_func, cov_func, lik_func, x, y);
        % obtain negative log margianl likelihood (nlZ)
        [nlZ, ~] = gp(hyp_opt, @infGaussLik, mean_func, cov_func, lik_func, x, y);
        grid(j,i) = nlZ;
    end
end

contourf(cov_vals, cov_vals, grid)
colormap(cool)
c = colorbar;
c.Label.String = 'Negative Log Marginal Likelihood';
xlabel('Log Length-Scale')
ylabel('Log Noise St Dev')
xticks(linspace(-5, 5, 11))
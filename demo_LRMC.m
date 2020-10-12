% simple demo for low rank matrix completion

num_experiments = 1; 

% definitions for experiments
m = 10;
n = m+100; 
oversampling_ratio = 3;
r = 3;
sigma_list = [1 1 1];

nv = floor(r*(n+m-r) * oversampling_ratio);    % number of observed entries


for counter =1:num_experiments
    % generate low rank matrix X with singular values in sigma_list 
    [X0, Utrue, Vtrue] = generate_low_rank_matrix(m,n,sigma_list); 

    % generate random set of observed entries
    t = randperm(n*m); 
    omega = zeros(nv,2);
    size_matrix = [m n]; 
    [omega(:,1), omega(:,2)] = ind2sub(size_matrix,t(1:nv)); 
    
    % generate matrix X with entries equal to X0(i,j) for (i,j) in omega
    X = zeros(m,n); 
    for i=1:nv
        X(omega(i,1),omega(i,2)) = X0(omega(i,1),omega(i,2));
    end
    
    % run R2RILS
    % (for options documentation, see opts_default in R2RILS_v2 function)
    opts.verbose = 1;
    opts.max_iter = 50;
    opts.LSQR_col_norm = 1;
    opts.init_option = 0;
    opts.weight_previous_estimate = 1.0 + sqrt(2);
    opts.early_stopping_RMSE_abs = 5e-14;
    opts.early_stopping_rel = 1e-7;
    opts.early_stopping_RMSE_rel = 5e-14;
    tic;
    [X_hat, U_hat, lambda_hat, V_hat, observed_RMSE, iter, convergence_flag] = R2RILS_v2(X, omega, r, opts); 
    %[X_hat, U_hat, lambda_hat, V_hat, observed_RMSE, iter, convergence_flag] = R2R_Beta(X, omega, r, 50, 1e-15); 
    elapsed_time = toc;
    
    % print results
    fprintf('experiment %4d RMSE %8d\n',counter,sqrt( sum(sum((X_hat - X0).^2)) ) / sqrt(n*m) ); 
    fprintf('TIME %5.1f\n',elapsed_time); 
end
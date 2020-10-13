% simple demo for low rank matrix completion

num_experiments = 50; 

% definitions for experiments
m = 1000;
n = 1000;
oversampling_ratio = 1.7;    % rho in the manuscript
r = 5;    %rank of matrix X_0 of size mxn
sigma_list = [10 8 4 2 1 ];    % singular values of matrix X_0

nv = floor(r*(n+m-r) * oversampling_ratio);    % number of observed entries

RMSE_R2R = zeros(num_experiments,1); 
ITER_R2R = zeros(num_experiments,1); 

for counter =1:num_experiments
    rng(counter+2020); 
    % generate low rank matrix X0 with singular values in sigma_list 
    [X0, Utrue, Vtrue] = generate_low_rank_matrix(m,n,sigma_list); 
    
    % generate mask and compute X accordingly
    p = nv/(m*n); 
    [omega, nv_actual, H] = generate_valid_mask(m,n,p,r,counter);
    X = H.* X0;
    
    % run R2RILS
    % (for options documentation, see opts_default in R2RILS function)
    opts.verbose = 1;
    opts.max_iter = 100;
    opts.LSQR_col_norm = 1;
    opts.init_option = 0;
    opts.weight_previous_estimate = 1.0 + sqrt(2);
    opts.early_stopping_RMSE_abs = 5e-14;
    opts.early_stopping_rel = 5e-11;
    opts.early_stopping_RMSE_rel = 5e-14;
    tic;
    [X_hat, U_hat, lambda_hat, V_hat, observed_RMSE, iter, convergence_flag] = ...
        R2RILS(X, omega, r, opts); 
    elapsed_time = toc;
    
    % sum results
    RMSE_R2R(counter) = sqrt( sum(sum((X_hat - X0).^2)) ) / sqrt(n*m);
    ITER_R2R(counter) = iter; 
    
    % print results
    fprintf('experiment %4d RMSE %8d\n',counter,RMSE_R2R(counter) ); 
    fprintf('TIME %5.1f\n',elapsed_time); 
    figure(3); clf; plot([1:counter],log10(sort(RMSE_R2R(1:counter))),'rs-'); grid on; title(counter); drawnow; 
end
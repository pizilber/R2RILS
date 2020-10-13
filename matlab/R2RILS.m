function [X_hat, U_hat, lambda_hat, V_hat, observed_RMSE, iter, convergence_flag] = ...
    R2RILS(X,omega,r,opts)
% function [X_hat, U_hat, lambda_hat, V_hat, observed_RMSE, iter, convergence_flag] = R2RILS_v2(X,omega,r,opts)
%
% WRITTEN BY BAUCH, NADLER & ZILBER / 2020
%
% INPUT: 
% X = matrix with observed entries in the set omega
% omega = list of size nv * 2 of pairs (i,j) ; nv=total number of visible entries
% r = target rank of reconstructed matrix
% opts: options meta-variable (see opts_default for details)
%

% default values for option variables
opts_default.verbose = 0;                       % display intermediate results
opts_default.max_iter = 50;                     % total number of iterations
opts_default.LSQR_col_norm = 0;                 % normalize columns of least squares matrix
opts_default.LSQR_max_iter = 4000;              % maximal number of iterations for the LSQR solver
opts_default.LSQR_tol = 1e-15;                  % tolerance of LSQR solver
opts_default.LSQR_smart_tol = 1;                % use LSQR_tol==observed_RMSE^2 when obersved_RMSE is low enough
opts_default.LSQR_smart_obj_min = 1e-5;         % observed_RMSE threshold to start using LSQR smart tol
opts_default.init_option = 0;                   % 0 for SVD, 1 for random, 2 for opts.init_U, opts.init_V
opts_default.init_U = NaN;                      % if opts.init_option==2, use this initialization for U
opts_default.init_V = NaN;                      % if opts.init_option==2, use this initialization for V
opts_default.weight_previous_estimate = 1.0;    % different averaging weight for previous estimate (should prevent oscillations)
opts_default.weight_from_iter = 40;             % start different averaging from itertion
opts_default.weight_every_iter = 7;             % use different averaging when mod(iter, weight_every_iter) < 2
opts_default.early_stopping_RMSE_abs = 5e-14;   % small observed_RMSE threshold (relevant to noise-free case). -1 for disabled
opts_default.early_stopping_rel = -1;           % small relative X_hat difference thrshold. -1 for disabled
opts_default.early_stopping_RMSE_rel = -1;      % small relative observed_RMSE difference. -1 for disabled

% for each unset option set its default value
fn = fieldnames(opts_default);
for k=1:numel(fn)
    if ~isfield(opts,fn{k}) || isempty(opts.(fn{k}))
        opts.(fn{k}) = opts_default.(fn{k});
        fn{k} = opts_default.(fn{k});
    end
end

% some definitions
[nr, nc] = size(X);   %nr,nc = number of rows / colums
m = (nc + nr) * r;  % total number of variables in single update iteration
nv = size(omega,1);   % number of observed entries
rhs = zeros(nv,1);   % vector of visible entries in matrix X
for counter=1:nv
    rhs(counter) = X(omega(counter,1),omega(counter,2)); 
end

% initialize U and V (of sizes nr x r and nc x r)
if opts.init_option == 0
    % initialization by rank-r SVD of observed matrix
    [U, ~, V] = svds(X,r);
elseif opts.init_option == 1
    % initialization by random orthogonal matrices
    Z = randn(nr,r);
    [U, ~, ~] = svd(Z,'econ'); 
    Z = randn(nc,r);
    [V, ~, ~] = svd(Z,'econ'); 
else
    % initiazliation by user-defined matrices
    U = opts.U_init;
    V = opts.V_init; 
end

X_hat_previous = zeros(nr,nc);   % previous intermediate rank 2r estimate
X_r_previous = zeros(nr,nc);   % rank-r projection of previous intermediate rank 2r estimate
observed_RMSE = zeros(opts.max_iter,1);   % of the rank-r projection of intermediate rank 2r matrix
X_max = max(max(abs(X)));
current_observed_RMSE = X_max;
best_RMSE = X_max; % initialized with max absolute value of all observed entries
convergence_flag = 0;

for loop_idx = 1:opts.max_iter
    
    if opts.verbose && (mod(loop_idx,5)==0)
        fprintf('INSIDE R2RILS-V2 loop_idx  %d/%d\n',loop_idx,opts.max_iter);
    end

    % Z^T = [a(coordinate 1)  a(coordinate 2) ... a(coordinate nc) | b(coordinate 1) ... b(coordinate nr) ]
    % Z^T = [ Vnew and then Unew ]
    Z = zeros(m,1);    % Z is a long vector with a vectorization of A and of B (notation in paper)
    A = zeros(nv,m);    % matrix of least squares problem
    B = zeros(1,m);    % used to normalize A columns if needed

    % contsruction of A and B
    for counter=1:nv
        j = omega(counter,1);
        k = omega(counter,2);
        index=r*(k-1)+1;
        A(counter,index:index+r-1) = U(j,:);
        B(index:index+r-1) = B(index:index+r-1) + U(j,:).^2;
        index = r*nc + r*(j-1)+1; 
        A(counter,index:index+r-1) = V(k,:);
        B(index:index+r-1) = B(index:index+r-1) + V(k,:).^2;
    end
    
    B = sqrt(B);
    if opts.LSQR_col_norm
        for j=1:size(A,2)
            A(:,j)=A(:,j)/B(j);
        end
    end
    A = sparse(A); 
    
    % determine tolerance for LSQR solver
    LSQR_tol = opts.LSQR_tol;
    if opts.LSQR_smart_tol
        LSQR_tol = min(opts.LSQR_smart_obj_min, current_observed_RMSE^2);
    end
    LSQR_tol = max(LSQR_tol, 2*eps);    % to supress warning
    % solve the least squares problem
    [Z, flag, relres_Z, iter_Z] = lsqr(A,rhs,LSQR_tol,opts.LSQR_max_iter);
        % LSQR finds the minimum norm solution and is much faster than lsqminnorm
    %disp([flag relres_Z iter_Z]);
    if opts.LSQR_col_norm
        Z = Z./B';
    end
    
    % construct Utilde and Vtilde from the entries of the long vector Z 
    Utilde = zeros(size(U)); Vtilde = zeros(size(V)); 
    nc_list = r * [0:1:(nc-1)]; 
    for i=1:r
        Vtilde(:,i) = Z(i+nc_list); 
    end
    nr_list = r * [0:1:(nr-1)]; 
    start_idx = r*nc; 
    for i=1:r
        Utilde(:,i) = Z(start_idx + i + nr_list);
    end
   
    % intermediate rank-2r estimate
    X_hat = U * Vtilde' + Utilde * V';
    
    % rank r projection of intermediate rank-2r solution
    [U_r, lambda_r, V_r] = svds(X_hat,r); 
    X_r = U_r * lambda_r * V_r'; 
    
    % calculate RMSE and update X_out if needed
    current_observed_RMSE = sqrt(sum(sum( (abs(X)>0).*(X_r - X).^2 )) / nv);
    observed_RMSE(loop_idx) = current_observed_RMSE; 
    if current_observed_RMSE < best_RMSE
        best_RMSE = current_observed_RMSE; 
        X_out = X_r; 
    end
    
    % col-normalize Utilde and Vtilde
    normU = sqrt(sum(Utilde.^2)); 
    Utilde = Utilde * diag(1./normU);  
    normV = sqrt(sum(Vtilde.^2)); 
    Vtilde = Vtilde * diag(1./normV);  

    % calculate next estimate of U, V by averaging previous estimate with
    % Utilde, Vtilde
    weight = 1.0; 
    if (loop_idx > opts.weight_from_iter && mod(loop_idx,opts.weight_every_iter)<2)         
        weight = opts.weight_previous_estimate;
        if opts.verbose
            fprintf('INSIDE R2RILS-V2 using different weight for previous estimate\n');
        end
    end
    U = (weight*U + Utilde);
    V = (weight*V + Vtilde); 
    
    % col-normalize U and V
    normU = sqrt(sum(U.^2)); 
    U = U * diag(1./normU);
    normV = sqrt(sum(V.^2)); 
    V = V * diag(1./normV); 

    % calculate new RMSE
    X_hat_diff = sqrt(sum(sum((X_hat-X_hat_previous).^2)))/sqrt(nr*nc);
    
    if opts.verbose
        fprintf('INSIDE R2RILS-V2 loop_idx %3d \t DIFF X_hat %6d DIFF X_r %6d observed_RMSE %6d\n',...
            loop_idx,X_hat_diff,...
            sqrt(sum(sum((X_r-X_r_previous).^2)))/sqrt(nr*nc), current_observed_RMSE);
        if loop_idx>1 
            fprintf('INSIDE R2RILS-V2 \t\t\t\t oberved_RMSE(new) / observed_RMSE(old)-1: %f\n',current_observed_RMSE/observed_RMSE(loop_idx-1)-1); 
        end
    end
    
    % early stopping criteria
    if current_observed_RMSE < opts.early_stopping_RMSE_abs
        if opts.verbose
            fprintf('INSIDE R2RILS-V2 Early stopping, small error on observed entries\n'); 
        end
        convergence_flag = 1;
    elseif X_hat_diff < opts.early_stopping_rel
        if opts.verbose
            fprintf('INSIDE R2RILS-V2 Early stopping, X_hat does not change\n'); 
        end
        convergence_flag = 1;
    elseif loop_idx > 1 && ...
            abs(observed_RMSE(loop_idx-1)/current_observed_RMSE-1) < opts.early_stopping_RMSE_rel
        if opts.verbose
            fprintf('INSIDE R2RILS-V2 Early stopping, observed_RMSE does not change\n'); 
        end
        convergence_flag = 1;
    end
    if convergence_flag
        break
    end
    
    X_hat_previous = X_hat;
    X_r_previous = X_r;
end

% return rank-r SVD of X_out, the rank-r matrix with the lowest observed
% RMSE during the iterations
[U_hat, lambda_hat, V_hat] = svds(X_out,r);
%X_hat = U_hat * lambda_hat * V_hat';
X_hat = X_out;
iter = loop_idx;
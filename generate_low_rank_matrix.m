function [X, U, V] = generate_low_rank_matrix(n_r,n_c,lambda)
%function [X U V] = generate_low_ram_matrix(n_r,n_c,lambda)
%
% INPUT: n_r,n_c = number of rows and columns
% lambda: non-zero singular values

% OUTPUT: X = n_r x n_c matrix of rank r
%         U = n_r x r left singular vectors
%         V = n_c x r right singular vectors

r = length(lambda); 
D = diag(lambda);    % diagonal rxr matrix
Z = randn(n_r,r); 

[U, t1, t2] = svd(Z,'econ'); 

Z = randn(n_c,r); 
[V, t1, t2] = svd(Z,'econ'); 

X = U * D * V'; 

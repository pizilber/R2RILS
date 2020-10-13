function [omega, nv, H] = generate_valid_mask(m,n,p,rank,random_seed)
%function [omega nv H] = generate_valid_mask(m,n,p,rank,random_seed)
%
% Generate random mask with probability p and with 
% at least r observed entries in each row and column

% initial random number generator
if nargin==5
    rng('default'); 
    rng(random_seed)
end

H = zeros(m,n); 
flag_mask = 1; 
while flag_mask
    H = (rand(m,n) < p);
    s1 = sum(H,1); 
    s2 = sum(H,2); 
    min1 = min(s1); 
    min2 = min(s2); 
    if min1 >= rank && min2 >= rank
        flag_mask = 0;    % reached a valid configuration, can exit the while loop
    end
    %fprintf('min %3d %3d\n',min1,min2); 
end

nv = sum(sum(H)); 
omega = zeros(nv,2); 
[omega(:,1), omega(:,2)] = find(H); 

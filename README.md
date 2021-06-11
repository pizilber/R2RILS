# R2RILS
This repository contains `Python` and `Matlab` implementations for `R2RILS` as described in J. Bauch, B. Nadler and P. Zilber (2021), available in [SIMODS](https://epubs.siam.org/doi/abs/10.1137/20M1315294) or [arXiv](https://arxiv.org/abs/2002.01849), as well as simple demos demonstrating the usage of the `Python` and `Matlab` implementations.
## Usage
#### Python
The entry point to run `R2RILS` is a function with the same name which expects the following parameters:
- required arguments:
  - X: input matrix to complete. This should be a numpy array of dimensions m x n.
  - omega: a mask matrix. 1 on observed entries, 0 otherwise.
  - rank: the target rank.
- optional arguments:
  - max_iter: maximal number of iterations.</li>
  - LSQR solver arguments:
    - lsqr_col_norm: if True, normalize columns of LSQR matrix.
    - lsqr_max_iter: maximal number of iterations of LSQR solver.
    - lsqr_tol: tolerance of LSQR solver.
    - lsqr_smart_tol: should increase LSQR's accuracy according to the current quality of the objective.
    - lsqr_smart_obj_min: minimal objective to start smart tolerance from.
  - initialization arguments:
    - init_option: 0 for SVD initialization, 1 for random, 2 for user-defined.
    - init_U: in case init_option==2, use this matrix to initialize U.
    - init_V: in case init_option==2, use this matrix to initialize V.
  - weight of previous estimate arguments:
    - weight_previous_estimate: different averaging weight for the previous estimate of U, V.
    - weight_from_iter: iteration number to start the different weighting from.
    - weight_every_iter: use different use different averaging when iter_num % weight_every_iter < 2.
  - early stopping arguments (see paper for exact definitions):
    - early_stopping_rmse_abs: eps for absolute difference between X_hat and X (RMSE), -1 for disabled.
    - early_stopping_rel: eps for relative difference of X_hat between iterations, -1 for disabled.
    - early_stopping_rmse_rel: eps for relative difference of RMSE between iterations, -1 for disabled.

This method returns X_hat - `R2RILS` estimate for X0, and a convergence flag which indicates if the algorithm converged.

#### Matlab
The entry point for running `R2RILS` in Matlab is again a function bearing the same name
which expects the following parameters:
- X: matrix with observed entries in the set omega.
- omega: array of pairs (i,j) indicating which entries are observed.
- rank: the target rank.
- opts: an optional meta-veraible, encapsulates the options detailed in the python implementation above.

This method returns [X_hat, U_hat, lambda_hat, V_hat, observed_RMSE, iter, convergence_flag] where:
- X_hat: rank r approximation of X0.
- U_hat: matrix of left singular vectors of X_hat.
- lambda_hat: singular values of X_hat.
- V_hat: matrix of right singular vectors of X_hat.
- observed_RMSE: the RMSE of X_hat on the observed entires of X0.
- iter: final iteration number of the algorithm.
- convergence_flag: incdicating whether the algorithm converged.

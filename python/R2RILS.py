"""
WRITTEN BY BAUCH, NADLER & ZILBER / 2020
"""


import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from sklearn.preprocessing import normalize

# initialization options
INIT_WITH_SVD = 0
INIT_WITH_RANDOM = 1
INIT_WITH_USER_DEFINED = 2

def R2RILS(X, omega, rank, verbose=True, max_iter=100,
           lsqr_col_norm=False, lsqr_max_iter=1000, lsqr_tol=1e-15,
           lsqr_smart_tol=True, lsqr_smart_obj_min=1e-5,
           init_option=INIT_WITH_SVD , init_U=None, init_V=None,
           weight_previous_estimate=1.0, weight_from_iter=40, weight_every_iter=7,
           early_stopping_rmse_abs=5e-14, early_stopping_rel=-1, early_stopping_rmse_rel=-1):
    """
    Run R2RILS algorithm.
    :param ndarray X: Input matrix (m,n). Unobserved entries should be zero.
    :param ndarray omega: Mask matrix (m,n). 1 on observed entries, 0 on unobserved.
    :param int rank: Underlying rank matrix.
    :param bool verbose: if True, display intermediate results.
    :param int max_iter: Maximal number of iterations to perform.
    :param bool lsqr_col_norm: if True, columns of the LSQR matrix are normalized.
    :param int lsqr_max_iter: max number of iterations fot the LSQR solver.
    :param float lsqr_tol: tolerance of LSQR solver.
    :param bool lsqr_smart_tol: if True, when objective <= lsqr_smart_obj_min, use lsqr_tol=objective**2.
    :param float lsqr_smart_obj_min: maximal obejctive to begin smart tolerance from.
    :param init_option: how to initialize U and V: 0 for SVD, 1 for random, 2 for user-defined.
    :param ndarray init_U: U initialization (m,rank), used in case init_option==2.
    :param ndarray init_V: V initialization (n,rank), used in case init_option==2.
    :param float weight_previous_estimate: different averaging weight for the previous estimate of U, V.
    :param int weight_from_iter: start the different weighting from this iteration number.
    :param int weight_every_iter: use the different weighting when iter_num % weight_every_iter < 2.
    :param float early_stopping_rmse_abs: eps for absolute difference between X_hat and X (RMSE), -1 for disabled.
    :param float early_stopping_rel: eps for relative difference of X_hat between iterations, -1 for disabled.
    :param float early_stopping_rmse_rel: eps for relative difference of RMSE between iterations, -1 for disabled.
    :return: R2RILS's estimate, and convergence flag (True if early steopped, False if iterations exceeded max).
    """
    m, n = X.shape
    num_visible_entries = np.count_nonzero(omega)

    # initial estimate
    if init_option == INIT_WITH_SVD:
      (U, _, V) = linalg.svds(X, k=rank, tol=1e-16)
      V = V.T
    elif init_option == INIT_WITH_RANDOM:
      U = np.random.randn(m, rank)
      V = np.random.randn(n, rank)
      U = np.linalg.qr(U)[0]
      V = np.linalg.qr(V)[0]
    else:
      U = init_U
      V = init_V
    # change the shapes from (m,r) and (n,r) to (r,m) and (r,n)
    U, V = U.T, V.T

    # generate sparse indices to accelerate future operations.
    sparse_matrix_rows, sparse_matrix_columns = generate_sparse_matrix_entries(omega, rank, m, n)
    # generate (constant) b for the least squares problem
    b = generate_b(X, omega, m, n)

    # start iterations
    early_stopping_flag = False
    X_hat_previous = np.dot(U.T, V)
    objective_previous = 1.
    best_RMSE = np.max(np.abs(X))
    X_hat_final = X_hat_previous  # X_hat_final will store the estimation with the best RMSE
    iter_num = 0
    objective = 1
    while iter_num < max_iter and not early_stopping_flag:
        if verbose and (iter_num % 5 == 0):
          print("iteration {}/{}".format(iter_num, max_iter))
        iter_num += 1
        
        # determine LSQR tolerance
        tol = lsqr_tol
        if lsqr_smart_tol:
          tol = min(lsqr_smart_obj_min, objective**2)
        
        # solve the least squares problem
        A = generate_sparse_A(U, V, omega, sparse_matrix_rows, sparse_matrix_columns, num_visible_entries, m, n,
                              rank)
        scale_vec = np.ones((A.shape[1]),)
        if lsqr_col_norm:
          A, scale_vec = rescale_A(A)
        x = linalg.lsqr(A, b, atol=tol, btol=tol, iter_lim=lsqr_max_iter)[0]
        x = x * scale_vec
        x = convert_x_representation(x, rank, m, n)
        
        # get new estimate for X_hat (project 2r onto r) and calculate its objective
        X_hat = get_estimated_value(x, U, V, rank, m, n)
        (U_r, Sigma_r, V_r) = linalg.svds(X_hat, k=rank, tol=1e-16)
        X_hat = U_r @ np.diag(Sigma_r) @ V_r
        objective = np.linalg.norm((X_hat - X) * omega, ord='fro') / np.sqrt(num_visible_entries)
        if verbose:
          print("objective: {}".format(objective))
        if objective < best_RMSE:
          X_hat_final = X_hat
          best_RMSE = objective

        # obtain new estimates for U and V
        U_tilde, V_tilde = get_U_V_from_solution(x, rank, m, n)
        # ColNorm U_tilde, V_tilde
        U_tilde = normalize(U_tilde, axis=1)
        V_tilde = normalize(V_tilde, axis=1)
        # average with previous estimate
        weight = 1.
        if (iter_num >= weight_from_iter) and (iter_num % weight_every_iter < 2):
          # this should prevent oscillations
          weight = weight_previous_estimate
          if verbose:
            print("using different weighting for previous estimate")
        U = normalize(weight * U + U_tilde, axis=1)
        V = normalize(weight * V + V_tilde, axis=1)

        # check early stopping criteria
        early_stopping_flag = False
        if early_stopping_rmse_abs > 0:
          early_stopping_flag |= objective < early_stopping_rmse_abs
        if early_stopping_rel > 0:
          early_stopping_flag |= np.linalg.norm(X_hat - X_hat_previous,
                                                ord='fro') / np.sqrt(m * n) < early_stopping_rel
        if early_stopping_rmse_rel > 0:
          early_stopping_flag |= np.abs(objective / objective_previous - 1) < early_stopping_rmse_rel

        # update previous X_hat and objective
        X_hat_previous = X_hat
        objective_previous = objective

    # return
    convergence_flag = iter_num < max_iter
    return X_hat_final, convergence_flag


def rescale_A(A):
    A = sparse.csc_matrix(A)
    scale_vec = 1. / linalg.norm(A, axis=0)
    return normalize(A, axis=0), scale_vec


def convert_x_representation(x, rank, m, n):
    recovered_x = np.array([x[k * rank + i] for i in range(rank) for k in range(n)])
    recovered_y = np.array([x[rank * n + j * rank + i] for i in range(rank) for j in range(m)])
    return np.append(recovered_x, recovered_y)


def generate_sparse_matrix_entries(omega, rank, m, n):
    row_entries = []
    columns_entries = []
    row = 0
    for j in range(m):
        for k in range(n):
            if 0 != omega[j][k]:
                # add indices for U entries
                for l in range(rank):
                    columns_entries.append(k * rank + l)
                    row_entries.append(row)
                # add indices for V entries
                for l in range(rank):
                    columns_entries.append((n + j) * rank + l)
                    row_entries.append(row)
                row += 1
    return row_entries, columns_entries


def generate_sparse_A(U, V, omega, row_entries, columns_entries, num_visible_entries, m, n, rank):
    U_matrix = np.array(U).T
    V_matrix = np.array(V).T
    # we're generating row by row
    data_vector = np.concatenate(
        [np.concatenate([U_matrix[j], V_matrix[k]]) for j in range(m) for k in range(n) if 0 != omega[j][k]])
    return sparse.csr_matrix(sparse.coo_matrix((data_vector, (row_entries, columns_entries)),
                                               shape=(num_visible_entries, rank * (m + n))))

def generate_b(X, omega, m, n):
    return np.array([X[j][k] for j in range(m)
                     for k in range(n) if 0 != omega[j][k]])


def get_U_V_from_solution(x, rank, m, n):
    V = np.array([x[i * n:(i + 1) * n] for i in range(rank)])
    U = np.array([x[rank * n + i * m: rank * n + (i + 1) * m] for i in range(rank)])
    return U, V


def get_estimated_value(x, U, V, rank, m, n):
    # calculate U's contribution
    estimate = np.sum(
        [np.dot(U[i].reshape(m, 1), np.array(x[i * n:(i + 1) * n]).reshape(1, n)) for i in
         range(rank)],
        axis=0)
    # calculate V's contribution
    estimate += np.sum(
        [np.dot(x[rank * n + i * m: rank * n + (i + 1) * m].reshape(m, 1),
                V[i].reshape(1, n))
         for i in range(rank)], axis=0)
    return estimate

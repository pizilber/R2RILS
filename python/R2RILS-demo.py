import numpy as np
from scipy.stats import ortho_group
from R2RILS import R2RILS,\
    INIT_WITH_SVD, INIT_WITH_RANDOM, INIT_WITH_USER_DEFINED

def run_demo():
    num_experiments = 1

    # experiment definitions
    m = 200
    n = 300
    rank = 3
    oversampling = 3.
    noise_level = 0
    singular_values = [1] * rank

    # algorithm options
    options = {
        # for documentation of these options, see the definition of R2RILS routine
        # general
        'verbose' : True,
        'max_iter' : 50,
        # least-squares parameters
        'lsqr_col_norm': False,
        'lsqr_max_iter': 1000,
        'lsqr_tol': 1e-15,
        'lsqr_smart_tol': True,
        'lsqr_smart_obj_min': 1e-5,
        # initialization
        'init_option': INIT_WITH_SVD,
        # weight of the previous estimate
        'weight_previous_estimate': 1.,
        'weight_from_iter': 40,
        'weight_every_iter': 7,
        # early stopping criteria
        'early_stopping_rmse_abs': 5e-14,
        'early_stopping_rel': -1,
        'early_stopping_rmse_rel': -1,
    }

    p = oversampling * rank * (m + n - rank) / (m * n)
    for i in range(num_experiments):
        U, V, omega, noise = generate_experiment_data(m, n, singular_values, p, noise_level)
        X0 = np.dot(U, V.T)
        X = X0 * omega + noise
        X_hat, _ = R2RILS(X, omega, rank, **options)
        num_visible_entries = np.count_nonzero(omega)
        unobserved_RMSE = np.linalg.norm((X_hat - X0) * (1 - omega), ord='fro') / np.sqrt(m * n - num_visible_entries)
        observed_RMSE = np.linalg.norm((X_hat - X0) * omega, ord='fro') / np.sqrt(num_visible_entries)
        print('experiment: {}, unobserved RMSE: {}, observed RMSE: {}'.format(i, unobserved_RMSE, observed_RMSE))


def generate_experiment_data(m, n, singular_values, p, noise_level=0, verbose=True):
    """
    Generate data for an experiment
    :param int m: number of rows in matrix.
    :param int n: number of columns in matrix.
    :param list singular_values: singular values.
    :param float p: probability of observing an entry.
    :param float noise_level: additive Gaussian noise standard deviation.
    :param bool verbose: display messages.
    :return: U, V, mask, noise such that
     - U x V.T is a rank len(singular_values) matrix with non zero singular values equal to singular_values.
     - omega is the matrix of observed entries, with 1 indicating that an entry is observed and 0 that it is not.
            omega is resampled until there are at least len(singular_values) visible entries in each column and row.
     - noise: an (m,n) matrix with i.i.d. Gaussian entries sampled with standard deviation noise_level.
    """
    rank = len(singular_values)
    U, V = generate_set_singular_values(m, n, rank, singular_values)
    # resample mask until there are enough measurements
    num_resamples = 0
    while True:
        num_resamples += 1
        omega = np.round((np.random.random((m, n)) + p) * 1. / 2)
        # count non zero on columns
        if min(np.count_nonzero(omega, axis=0)) < rank:
            if verbose and (num_resamples % 100 == 0):
              print('resampling mask {}'.format(num_resamples))
            continue
        if min(np.count_nonzero(omega, axis=1)) < rank:
            if min(np.count_nonzero(omega, axis=0)) < rank:
              if verbose and (num_resamples % 100 == 0):
                print('resampling mask {}'.format(num_resamples))
            continue
        break
    noise = np.random.randn(m, n) * noise_level
    return U, V, omega, noise


def generate_set_singular_values(m, n, rank, singular_values):
    U = np.random.randn(m, rank)
    V = np.random.randn(n, rank)
    U = np.linalg.qr(U)[0]
    V = np.linalg.qr(V)[0]
    return U, singular_values * V


if __name__ == '__main__':
    run_demo()

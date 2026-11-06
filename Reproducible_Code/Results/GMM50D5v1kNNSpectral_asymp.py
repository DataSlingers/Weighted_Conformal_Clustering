import sys
sys.path.append('../../')

import numpy as np  
from weighted_conformal_clustering.utils import WeightedConformalClustering, ConformalClustering, GMMCutoff, sample_gmm

import pickle
import copy
import joblib
from joblib import Parallel, delayed


K = 5
d = 50

# generate means in 2D by equispaced points on a circle
means_2d = 4 * np.vstack([np.cos(np.pi * np.linspace(0, 2, K, endpoint=False)),
                      np.sin(np.pi * np.linspace(0, 2, K, endpoint=False))]).T

# embedding the 2D means into higher dimensions by padding with zeros
means = np.zeros((K, d))
means[:, :2] = means_2d

base_covariances = np.array([np.eye(d) for _ in range(K)])
weights = np.ones(K) / K


def run_experiment(n, var, seed):
    covariances = base_covariances * var
    n_al = np.min([100, n//10]) # 10% or 100 points for alignment
    n_total = n + n_al + 1 # total points needed
    X, y = sample_gmm(means, covariances, weights, n_samples=n_total, seed=seed)

    n_tr = int(0.5*n)
    X_tr, y_tr = X[:n_tr], y[:n_tr] # training
    X_cal, y_cal = X[n_tr:n], y[n_tr:n] # calibration
    X_al, y_al = X[n:(n+n_al)], y[n:(n+n_al)] # alignment
    X_val, y_val = X[(n+n_al):(n+n_al+1)], y[(n+n_al):(n+n_al+1)] # validation

    # weighted conformal clustering with SpectralGMMS + kNN classifier
    wcc = WeightedConformalClustering(X_tr, X_cal)
    wcc.set_classifier('kNN', n_neighbors=3)
    # repeat training until K clusters are found (change seed)
    found_K = False
    attempt = 0
    while not found_K:
        try:
            wcc.train('SpectralGMMS', embedding='svd', n_components=K, max_iter=200, n_init=5, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    conformal_set = wcc.conformal_set(X_val, 'SpectralGMMS', embedding='svd', n_components=K, max_iter=200, n_init=5, random_state=0)
    res_W = wcc.validate(y_val, conformal_set, X_al, y_al)

    # conformal clustering with SpectralGMMS + kNN classifier
    cc = ConformalClustering(X_tr, X_cal)
    clf = copy.deepcopy(wcc.clf) # use the same classifier for fair comparison
    cc.set_classifier('pretrain', clf=clf)
    cc.fit('SpectralGMMS', embedding='svd', n_components=K, max_iter=200, n_init=5, random_state=0)
    res_S = cc.validate(X_val, y_val, X_al, y_al, method='narc')

    results = {
        'n': n,
        'var': var,
        'seed': seed,
        'WeightedCC': res_W,
        'SplitCC': res_S,
    }

    return results


if __name__ == '__main__':
    var = 2.3
    B = 1000
    np.random.seed(0)
    seeds = np.random.permutation(10*B)[:B]
    n_range = [300, 500, 800, 1200, 2000, 3000, 5000, 7500]

    print(f'Available CPUs by joblib: {joblib.cpu_count()}')
    results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_experiment)(n, var, seed)
            for n in n_range
            for seed in seeds
        )

    with open(f'GMM50D5v1kNNSpectral_asymp.pkl', 'wb') as f:
        pickle.dump(results, f)

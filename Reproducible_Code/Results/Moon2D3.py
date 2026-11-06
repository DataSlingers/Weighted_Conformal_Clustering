import sys
sys.path.append('../../')

import numpy as np
from weighted_conformal_clustering.utils import WeightedConformalClustering, ConformalClustering, gen_moons3

import pickle
import copy
import joblib
from joblib import Parallel, delayed


K = 3


def run_experiment(n, var, seed):
    n_al = np.min([100, n//10]) # 10% or 100 points for alignment
    n_total = n + n_al + 1 # total points needed
    X, y = gen_moons3(n_samples=n_total, noise=var*0.05, random_state=seed)

    n_tr = int(0.5*n)
    n_cal = n - n_tr
    X_tr, y_tr = X[:n_tr], y[:n_tr] # training
    X_cal, y_cal = X[n_tr:n], y[n_tr:n] # calibration
    X_al, y_al = X[n:(n+n_al)], y[n:(n+n_al)] # alignment
    X_val, y_val = X[(n+n_al):(n+n_al+1)], y[(n+n_al):(n+n_al+1)] # validation

    # weighted conformal clustering with SpectralGMMS + kNN classifier
    wcc = WeightedConformalClustering(X_tr, X_cal)
    wcc.set_classifier('kNN', n_neighbors=int(np.sqrt(n_tr/2)))
    found_K = False
    attempt = 0
    while not found_K:
        try:
            wcc.train('SpectralGMMS', n_neighbors=int(n_tr*0.05), n_components=K, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    conformal_set = wcc.conformal_set(X_val, 'SpectralGMMS', n_neighbors=int((n_cal+1)*0.05), n_components=K, random_state=attempt)
    res_W = wcc.validate(y_val, conformal_set, X_al, y_al)

    # conformal clustering with SpectralGMMS + classifier
    cc = ConformalClustering(X_tr, X_cal)
    clf = copy.deepcopy(wcc.clf) # use the same classifier for fair comparison
    cc.set_classifier('pretrain', clf=clf)
    cc.fit('SpectralGMMS', n_neighbors=int(n_tr*0.05), n_components=K, random_state=attempt)
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
    n = 600
    B = 1000
    np.random.seed(0)
    seeds = np.random.permutation(10*B)[:B]
    var_range = np.linspace(1, 2.4, 8)

    print(f'Available CPUs by joblib: {joblib.cpu_count()}')
    results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_experiment)(n, var, seed)
            for var in var_range
            for seed in seeds
        )

    with open(f'Moon2D3.pkl', 'wb') as f:
        pickle.dump(results, f)
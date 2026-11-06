import sys
sys.path.append('../../')

import numpy as np  
from weighted_conformal_clustering.utils import WeightedConformalClustering, ConformalClustering, GMMCutoff, sample_gmm, soft_predict, label_alignment, scale_weights, conformal_narc

from sklearn.mixture import GaussianMixture

import pickle
import copy
import joblib
from joblib import Parallel, delayed


K = 5
d = 2

# generate means in 2D by equispaced points on a circle
means = 4 * np.vstack([np.cos(np.pi * np.linspace(0, 2, K, endpoint=False)),
                      np.sin(np.pi * np.linspace(0, 2, K, endpoint=False))]).T

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

    # weighted conformal clustering with GMMS + kNN classifier
    wcc = WeightedConformalClustering(X_tr, X_cal)
    wcc.set_classifier('kNN', n_neighbors=int(np.sqrt(n_tr)))
    # repeat training until K clusters are found (change seed)
    found_K = False
    attempt = 0
    while not found_K:
        try:
            wcc.train('GMMS', n_components=K, max_iter=200, n_init=5, random_state=attempt)
            found_K = True
        except ValueError:
            attempt += 1
    conformal_set = wcc.conformal_set(X_val, 'GMMS', n_components=K, max_iter=200, n_init=5, random_state=0)
    res_W = wcc.validate(y_val, conformal_set, X_al, y_al)

    # weighted conformal clustering without augmentation
    clst_cal = GaussianMixture(n_components=K, max_iter=200, n_init=5, random_state=0)
    clst_cal.fit(X_cal)
    clst_cal_prob = clst_cal.predict_proba(X_cal)
    y_cal_clst = soft_predict(clst_cal_prob)
    clst_cal_weight = clst_cal_prob[np.arange(len(X_cal)), y_cal_clst]
    clst_test_prob = clst_cal.predict_proba(X_val) # shape = (1, K)
    clst_test_weight = clst_test_prob[0, :] # (K,) weights for the test point based on the clustering probabilities (GMM fitted on the calibration data only)

    # reuse the classifier trained by wcc
    clf = copy.deepcopy(wcc.clf) # use the same classifier for fair comparison
    cal_prob = clf.predict_proba(X_cal) # (n_cal, K)
    test_prob = clf.predict_proba(X_val) # (1, K)

    # align calibration labels using the calibration data only (most probable clusters)
    y_cal_hat = clf.predict(X_cal) # (n_cal,)
    align_to_tr = label_alignment(y_cal_hat, np.argmax(clst_cal_prob, axis=1), K)
    y_cal_al = align_to_tr[y_cal_clst] # aligned calibration labels (still the sampled labels)

    # calculate the weights (same clipping as in WeightedConformalClustering.conformal_set)
    clip_min = 1e-6
    weight_cal_numerator = cal_prob[np.arange(len(X_cal)), y_cal_al]
    weight_test_numerator = test_prob # (1, K)
    weight_cal_raw = weight_cal_numerator / np.maximum(clst_cal_weight, clip_min)
    weight_cal_raw[(weight_cal_numerator < clip_min) & (clst_cal_weight < clip_min)] = 1.0

    # without augmentation, using a one-hot clustering weight for the test point (hard assignment to its most probable cluster)
    clst_test_weight_OH = np.zeros(K)
    clst_test_weight_OH[align_to_tr[np.argmax(clst_test_weight)]] = 1.0 # (K,)
    weight_test_raw_OH = weight_test_numerator / np.maximum(clst_test_weight_OH, clip_min)
    weight_test_raw_OH[(weight_test_numerator < clip_min) & (clst_test_weight_OH < clip_min)] = 1.0
    weight_OH = scale_weights(weight_cal_raw, weight_test_raw_OH) # (n_cal, 1, K)
    conformal_set_bool = conformal_narc(cal_prob, y_cal_al, test_prob, alpha=wcc.alpha, weight=weight_OH) # (1, K)
    res_NA_OH = wcc.validate(y_val, np.where(conformal_set_bool[0])[0], X_al, y_al)

    results = {
        'n': n,
        'var': var,
        'seed': seed,
        'WeightedCC': res_W,
        'WeightedCC_NA_OH': res_NA_OH,
    }

    return results


if __name__ == '__main__':
    var = 2.6
    B = 1000
    np.random.seed(0)
    seeds = np.random.permutation(10*B)[:B]
    n_range = [200, 300, 500, 800, 1200, 1800, 2700, 4000]

    print(f'Available CPUs by joblib: {joblib.cpu_count()}')
    results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_experiment)(n, var, seed)
            for n in n_range
            for seed in seeds
        )

    with open(f'ablation_GMM2D5v1kNN_asymp.pkl', 'wb') as f:
        pickle.dump(results, f)

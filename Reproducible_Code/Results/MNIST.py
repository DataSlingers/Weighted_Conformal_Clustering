import sys
sys.path.append('../../')

import numpy as np
from weighted_conformal_clustering.utils import label_alignment, soft_predict, conformal_narc, scale_weights

import pickle
from joblib import Parallel, delayed

from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, StratifiedKFold


K = 10


if __name__ == '__main__':
    with open("MNIST_10D.pkl", "rb") as f:
        data = pickle.load(f)
        X_10d = data['X_10d']
        y_true = data['y_true']

    # first obtaine 10 folds for stratified cross-validation, then run the experiment on each fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(skf.split(X_10d, y_true))

    results = []
    for fold_idx, (tr_cal_idx, val_idx) in enumerate(folds):
        print(f"Running fold {fold_idx+1}/10...")
        X_fold, y_fold = X_10d[tr_cal_idx], y_true[tr_cal_idx]
        X_tr, X_cal, y_tr, y_cal = train_test_split(X_fold, y_fold, test_size=0.5, random_state=fold_idx, stratify=y_fold)
        X_val, y_val = X_10d[val_idx], y_true[val_idx]

        n_tr = X_tr.shape[0]
        n_cal = X_cal.shape[0]
        n_val = X_val.shape[0]

        clf = KNeighborsClassifier(n_neighbors=int(np.sqrt(n_tr)))
        gmm_tr = GaussianMixture(n_components=K, max_iter=200, n_init=5, random_state=0)
        gmm_tr.fit(X_tr)
        gmm_tr_prob = gmm_tr.predict_proba(X_tr)
        y_tr_gmm = soft_predict(gmm_tr_prob, seed=0)
        clf.fit(X_tr, y_tr_gmm)

        cal_prob = clf.predict_proba(X_cal)
        y_cal_hat = clf.predict(X_cal)
        
        # Pre-fit a GMM on X_cal to use for warm-starting the augmented GMMs.
        # This saves an enormous amount of computation time since we avoid fitting from scratch per validation point.
        gmm_cal = GaussianMixture(n_components=K, max_iter=200, n_init=5, random_state=0)
        gmm_cal.fit(X_cal)

        def conformal_set(x, alpha=0.1):
            X_aug = np.vstack([X_cal, x]) # x has to be (1, n_features) for vstack to work
            
            # Warm start using the calibration GMM parameters, reducing n_init to 1
            gmm_aug = GaussianMixture(
                n_components=K, 
                max_iter=200, 
                n_init=1, 
                random_state=0,
                weights_init=gmm_cal.weights_,
                means_init=gmm_cal.means_,
                precisions_init=gmm_cal.precisions_
            )
            gmm_aug.fit(X_aug)
            gmm_aug_prob = gmm_aug.predict_proba(X_aug)
            gmm_cal_prob = gmm_aug_prob[:n_cal, :]
            y_cal_gmm = soft_predict(gmm_cal_prob, seed=0)
            gmm_cal_weight = gmm_cal_prob[np.arange(n_cal), y_cal_gmm]
            gmm_test_weight = gmm_aug_prob[-1, :]

            test_prob = clf.predict_proba(x)

            
            align_to_tr = label_alignment(y_cal_hat, y_cal_gmm, K=K)
            y_cal_al = align_to_tr[y_cal_gmm]

            weight_cal_numerator = cal_prob[np.arange(n_cal), y_cal_al]
            weight_cal_raw = weight_cal_numerator / (gmm_cal_weight + 1e-15)

            weight_test_numerator = test_prob[:, align_to_tr]
            weight_test_raw = weight_test_numerator / (gmm_test_weight + 1e-15)

            weight = scale_weights(weight_cal_raw, weight_test_raw)

            conformal_set_bool = conformal_narc(cal_prob, y_cal_al, test_prob, weight=weight, alpha=alpha)
            return np.where(conformal_set_bool[0])[0]


        conformal_sets = Parallel(n_jobs=-1, verbose=1)(delayed(conformal_set)(X_val[i].reshape(1, -1)) for i in range(n_val))

        # align 
        oracle_alignment = label_alignment(y_tr, y_tr_gmm, K=K)
        conformal_sets_aligned = [oracle_alignment[cs] for cs in conformal_sets]

        coverage = np.mean([y_val[i] in conformal_sets_aligned[i] for i in range(len(X_val))])

        size = np.mean([len(cs) for cs in conformal_sets_aligned])

        results.append((coverage, size))

    print("")

    with open("MNIST.pkl", "wb") as f:
        pickle.dump(results, f)
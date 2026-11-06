# Import clustering models
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
import skfuzzy as fuzz

# Import classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import other utilities
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_array, csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, svd
from scipy.special import gammaln, psi, polygamma, logsumexp
from scipy.stats import multivariate_normal
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import kneighbors_graph
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import make_moons


def weighted_quantile(cal_scores, weight, u):
    """
    Computes the u-th quantile for m x K different weighted empirical measures over the same scores, accounting for probability mass at infinity = quantiles of sum_i w_i delta_{score_i} + (1 - sum_i w_i) delta_{+inf}.
    
    Parameters:
    - cal_scores: 1D numpy array of shape (n,)
    - weight: 3D numpy array of shape (n, m, K)
    - u: float, the target quantile in (0, 1)
    
    Returns:
    - 2D numpy array of shape (m, K) containing the quantiles.
    """
    # 1. Sort scores. Apply the same sorting to the first dimension of weights.
    # The Ellipsis (...) ensures we slice axis 0 while keeping m and K intact.
    sort_idx = np.argsort(cal_scores)
    sorted_scores = cal_scores[sort_idx]
    sorted_weights = weight[sort_idx, ...] 
    
    # 2. Cumulative sum down the first axis (n)
    cdf_steps = np.cumsum(sorted_weights, axis=0)
    
    # 3. Boolean mask where cumulative sum meets or exceeds u
    mask = cdf_steps >= u
    
    # 4. Find the first True value along axis 0. 
    # This collapses the (n, m, K) mask into an (m, K) array of indices.
    idx = np.argmax(mask, axis=0)
    
    # 5. Handle the infinity condition (if u > total sum of a specific m, K slice)
    # Check the very last slice of the n-dimension
    valid = mask[-1, ...] 
    
    # 6. Construct the (m, K) result array
    m, K = weight.shape[1], weight.shape[2]
    quantiles = np.full((m, K), np.inf)
    
    # Advanced boolean indexing: 
    # idx[valid] extracts the integer indices only where valid is True.
    # sorted_scores gets the corresponding score for those indices.
    quantiles[valid] = sorted_scores[idx[valid]]
    
    return quantiles

def conformal_arc(cal_prob, Y_cal, test_prob, weight=None, alpha=0.1):
    """
    Based on Romano et al. (2020) for classification - adaptive and reliable classification (ARC)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # sort the calibration probabilities
    sorted_cal_prob_ids = np.argsort(-cal_prob, axis=1) # descending order (r inverse in my notes)
    sorted_cal_probs = np.take_along_axis(cal_prob, sorted_cal_prob_ids, axis=1) # sorted probabilities
    sorted_cal_probs_cumsum = np.cumsum(sorted_cal_probs, axis=1) # cumulative sum of sorted probabilities
    
    # ranking (map r() in my notes)
    cal_prob_ranks = np.argsort(sorted_cal_prob_ids, axis=1) # rank of each class in each calibration sample

    # compute the ranks corresponding to the labels in the calibration set
    rank_Y_cal = cal_prob_ranks[np.arange(n_cal), Y_cal] # rank of the true class in each calibration sample

    # randomization
    U_cal = np.random.random_sample(n_cal)  # uniform random numbers for each calibration sample
    
    # scores (E in the paper)
    cal_scores = sorted_cal_probs_cumsum[np.arange(n_cal), rank_Y_cal] - sorted_cal_probs[np.arange(n_cal), rank_Y_cal]*U_cal

    # get quantiles
    if weight is not None:
        # weight = (n_cal, n_test, K)
        q = weighted_quantile(cal_scores, weight, 1 - alpha) # (n_test, K) array of quantiles
    else:
        sorted_cal_scores = np.sort(cal_scores)
        q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
        if q_rank > n_cal:
            q = np.inf  # handle the case where q_rank exceeds the number of calibration samples
        else:
            q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # test scores
    sorted_test_prob_ids = np.argsort(-test_prob, axis=1) # descending order
    sorted_test_probs = np.take_along_axis(test_prob, sorted_test_prob_ids, axis=1) # sorted probabilities
    sorted_test_probs_cumsum = np.cumsum(sorted_test_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking 
    test_prob_ranks = np.argsort(sorted_test_prob_ids, axis=1) # rank of each class in each test sample

    # randomization
    U_test = np.random.random_sample(n_test)  # uniform random numbers for each test sample
    test_scores = sorted_test_probs_cumsum - sorted_test_probs*U_test[:, np.newaxis] # each row is ascending

    # compute the conformal set
    # prediction_set_rank = test_scores <= q # based on the ranks
    # prediction_set = np.take_along_axis(prediction_set_rank, test_prob_ranks, axis=1)   # rank to the original indices
    test_scores_label = np.take_along_axis(test_scores, test_prob_ranks, axis=1)   # rank to the original indices: column j = score of label j
    prediction_set = test_scores_label <= q   # label j vs q[:, j]; q is a scalar when weight is None

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector

def conformal_arc_ne(cal_prob, Y_cal, test_prob, weight=None, alpha=0.1):
    """
    conformal_arc with non-empty guarantee (include the most probable label if empty)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # sort the calibration probabilities
    sorted_cal_prob_ids = np.argsort(-cal_prob, axis=1) # descending order (r inverse in my notes)
    sorted_cal_probs = np.take_along_axis(cal_prob, sorted_cal_prob_ids, axis=1) # sorted probabilities
    sorted_cal_probs_cumsum = np.cumsum(sorted_cal_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking (map r() in my notes)
    cal_prob_ranks = np.argsort(sorted_cal_prob_ids, axis=1) # rank of each class in each calibration sample

    # compute the ranks corresponding to the labels in the calibration set
    rank_Y_cal = cal_prob_ranks[np.arange(n_cal), Y_cal] # rank of the true class in each calibration sample

    # randomization
    U_cal = np.random.random_sample(n_cal)  # uniform random numbers for each calibration sample

    # scores (E in the paper)
    cal_scores = sorted_cal_probs_cumsum[np.arange(n_cal), rank_Y_cal] - sorted_cal_probs[np.arange(n_cal), rank_Y_cal]*U_cal

    # get quantiles
    if weight is not None:
        # weight = (n_cal, n_test, K)
        q = weighted_quantile(cal_scores, weight, 1 - alpha) # (n_test, K) array of quantiles
    else:
        sorted_cal_scores = np.sort(cal_scores)
        q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
        if q_rank > n_cal:
            q = np.inf  # handle the case where q_rank exceeds the number of calibration samples
        else:
            q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # test scores
    sorted_test_prob_ids = np.argsort(-test_prob, axis=1) # descending order
    sorted_test_probs = np.take_along_axis(test_prob, sorted_test_prob_ids, axis=1) # sorted probabilities
    sorted_test_probs_cumsum = np.cumsum(sorted_test_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking
    test_prob_ranks = np.argsort(sorted_test_prob_ids, axis=1) # rank of each class in each test sample

    # randomization
    U_test = np.random.random_sample(n_test)  # uniform random numbers for each test sample
    test_scores = sorted_test_probs_cumsum - sorted_test_probs*U_test[:, np.newaxis] # each row is ascending

    # compute the conformal set
    test_scores_label = np.take_along_axis(test_scores, test_prob_ranks, axis=1)   # rank to the original indices: column j = score of label j
    prediction_set = test_scores_label <= q   # label j vs q[:, j]; q is a scalar when weight is None
    # prediction_set_rank = test_scores <= q # based on the ranks
    # prediction_set = np.take_along_axis(prediction_set_rank, test_prob_ranks, axis=1)   # rank to the original indices

    # ensure non-empty sets (include the most probable label if empty)
    most_probable_labels = np.argmax(test_prob, axis=1)
    prediction_set[np.arange(n_test), most_probable_labels] = True
    
    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector

def conformal_narc(cal_prob, Y_cal, test_prob, weight=None, alpha=0.1):
    """
    conformal_arc without randomization (most probable label always included)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # sort the calibration probabilities
    sorted_cal_prob_ids = np.argsort(-cal_prob, axis=1) # descending order (r inverse in my notes)
    sorted_cal_probs = np.take_along_axis(cal_prob, sorted_cal_prob_ids, axis=1) # sorted probabilities
    sorted_cal_probs_cumsum = np.cumsum(sorted_cal_probs, axis=1) # cumulative sum of sorted probabilities
    
    # ranking (map r() in my notes)
    cal_prob_ranks = np.argsort(sorted_cal_prob_ids, axis=1) # rank of each class in each calibration sample

    # compute the ranks corresponding to the labels in the calibration set
    rank_Y_cal = cal_prob_ranks[np.arange(n_cal), Y_cal] # rank of the true class in each calibration sample

    # scores (E in the paper)
    cal_scores = sorted_cal_probs_cumsum[np.arange(n_cal), rank_Y_cal] - sorted_cal_probs[np.arange(n_cal), rank_Y_cal]

    # get quantiles
    if weight is not None:
        # weight = (n_cal, n_test, K)
        q = weighted_quantile(cal_scores, weight, 1 - alpha) # (n_test, K) array of quantiles
    else:
        sorted_cal_scores = np.sort(cal_scores)
        q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
        if q_rank > n_cal:
            q = np.inf  # handle the case where q_rank exceeds the number of calibration samples
        else:
            q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # test scores
    sorted_test_prob_ids = np.argsort(-test_prob, axis=1) # descending order
    sorted_test_probs = np.take_along_axis(test_prob, sorted_test_prob_ids, axis=1) # sorted probabilities
    sorted_test_probs_cumsum = np.cumsum(sorted_test_probs, axis=1) # cumulative sum of sorted probabilities

    # ranking 
    test_prob_ranks = np.argsort(sorted_test_prob_ids, axis=1) # rank of each class in each test sample
    test_scores = sorted_test_probs_cumsum - sorted_test_probs # each row is ascending

    # compute the conformal set
    # prediction_set_rank = test_scores <= q # based on the ranks
    # prediction_set = np.take_along_axis(prediction_set_rank, test_prob_ranks, axis=1)   # rank to the original indices
    test_scores_label = np.take_along_axis(test_scores, test_prob_ranks, axis=1)   # rank to the original indices: column j = score of label j
    prediction_set = test_scores_label <= q   # label j vs q[:, j]; q is a scalar when weight is None

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector

def conformal_clp(cal_prob, Y_cal, test_prob, weight=None, alpha=0.1):
    """
    Compute the conformal set using the classification probability
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # compute the classification scores = pi_hat(Y_i | X_i) for i in cal
    # conformal scores = negative probabilities
    cal_scores = -cal_prob[np.arange(n_cal), Y_cal]

    # get quantiles
    if weight is not None:
        # weight = (n_cal, n_test, K)
        q = weighted_quantile(cal_scores, weight, 1 - alpha) # (n_test, K) array of quantiles
    else:
        sorted_cal_scores = np.sort(cal_scores)
        q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
        if q_rank > n_cal:
            q = np.inf  # handle the case where q_rank exceeds the number of calibration samples
        else:
            q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # compute the conformal set
    test_scores = -test_prob
    prediction_set = test_scores <= q

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector

def conformal_clp_ne(cal_prob, Y_cal, test_prob, weight=None, alpha=0.1):
    """
    conformal_clp with non-empty guarantee (include the most probable label if empty)
        cal_prob: calibration probabilities (n_cal, K)
        Y_cal: calibration labels (n_cal,)
        test_prob: test probabilities (n_test, K)
    """
    n_cal, K = cal_prob.shape
    n_test = test_prob.shape[0]

    # compute the classification scores = pi_hat(Y_i | X_i) for i in cal
    # conformal scores = negative probabilities
    cal_scores = -cal_prob[np.arange(n_cal), Y_cal]

    # get quantiles
    if weight is not None:
        # weight = (n_cal, n_test, K)
        q = weighted_quantile(cal_scores, weight, 1 - alpha) # (n_test, K) array of quantiles
    else:
        sorted_cal_scores = np.sort(cal_scores)
        q_rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
        if q_rank > n_cal:
            q = np.inf  # handle the case where q_rank exceeds the number of calibration samples
        else:
            q = sorted_cal_scores[q_rank-1] # q_rank-th smallest value among the scores

    # compute the conformal set
    test_scores = -test_prob
    prediction_set = test_scores <= q

    # ensure non-empty sets (include the most probable label if empty)
    most_probable_labels = np.argmax(test_prob, axis=1)
    prediction_set[np.arange(n_test), most_probable_labels] = True

    return prediction_set   # (n_test, K) with each row = prediction set as a boolean vector

def label_alignment(t, s, K):
    """
    Get the alignment between two labels t = (t_1, ..., t_n) and s = (s_1, ..., s_n)
    
    Parameters:
    - t: array of true labels (n,)
    - s: array of obtained labels (n,)
    - K: number of clusters
    
    Returns:
    - sigma: desired permutation such that t_i = sigma(s_i) as many i's as possible
    """    

    if len(t) != len(s):
        raise ValueError("Labels must have the same length.")
    
    n = len(t)
    all_ones = np.ones(n)
    arange_n = np.arange(n)
    
    # create the sparse matrices 
    t_matrix = csr_array((all_ones, (arange_n, t)), shape=(n, K))  # rows = e_{t_i}
    s_matrix = csr_array((all_ones, (arange_n, s)), shape=(n, K))  # rows = e_{s_i}

    # solve the linear assignment problem
    Q = -s_matrix.T @ t_matrix # (K, K) matrix = small matrix
    Q = Q.toarray()  # convert to dense array for linear_sum_assignment ()
    _, col_ind = linear_sum_assignment(Q)

    return col_ind

def sample_gmm(means, covariances, weights, n_samples, seed=None):
    """
    Efficiently samples from a Gaussian Mixture Model.

    Parameters:
    - means: (K, D) array of component means.
    - covariances: (K, D, D) array of component covariance matrices.
    - weights: (K,) array of component mixing proportions (must sum to 1).
    - n_samples: int, number of samples to generate.
    - seed: int or None, random seed for reproducibility.

    Returns:
    - X: (n_samples, D) array of generated samples.
    - y: (n_samples,) array of component labels (0 to K-1) corresponding to X.
    """
    rng = np.random.default_rng(seed)
    
    # Ensure inputs are numpy arrays
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    weights = np.asarray(weights)
    
    # Validate shapes
    K, D = means.shape
    
    # 1. Determine how many samples to draw from each component
    # This vectorizes the component selection process
    n_samples_per_component = rng.multinomial(n_samples, weights)
    
    # Pre-allocate memory for speed
    X = np.empty((n_samples, D))
    y = np.empty(n_samples, dtype=int)
    
    # 2. Generate samples for each component in batches
    current_idx = 0
    for k, count in enumerate(n_samples_per_component):
        if count > 0:
            # Generate 'count' samples efficiently using optimized C-backend
            samples = rng.multivariate_normal(means[k], covariances[k], size=count)
            
            # Fill the pre-allocated arrays
            X[current_idx : current_idx + count] = samples
            y[current_idx : current_idx + count] = k
            
            current_idx += count
            
    # 3. Shuffle the result to mix components randomly
    # (Otherwise, output is sorted by component 0, then 1, etc.)
    perm = rng.permutation(n_samples)
    
    return X[perm], y[perm]

def sample_gamm(shapes, scales, weights, n_samples, seed=None):
    """
    Efficiently sample from a Gamma Mixture Model.

    Parameters:
    - shapes : (K, D) array of shapes (alpha) per component and dimension. Must be > 0.
    - scales : (K, D) array of scales (theta) per component and dimension. Must be > 0.
    - weights : (K,) array of component mixing proportions (must sum to 1)
    - n_samples : int, number of samples to generate.
    - seed : int or None, random seed for reproducibility.

    Returns:
    - X : (n_samples, D) array of generated samples.
    - y : (n_samples,) array of component labels (0 to K-1) corresponding to X.
    """
    rng = np.random.default_rng(seed)

    # Ensure inputs are numpy arrays
    shapes = np.asarray(shapes)
    scales = np.asarray(scales)
    weights = np.asarray(weights)

    # Validate shapes
    K, D = shapes.shape

    # 1. Determine how many samples to draw from each component
    # This vectorizes the component selection process
    n_samples_per_component = rng.multinomial(n_samples, weights)

    # Pre-allocate memory for speed
    X = np.empty((n_samples, D))
    y = np.empty(n_samples, dtype=int)

    # 2. Generate samples for each component in batches
    current_idx = 0
    for k, count in enumerate(n_samples_per_component):
        if count > 0:
            # Generate 'count' samples efficiently using optimized C-backend
            samples = rng.gamma(shape=shapes[k], scale=scales[k], size=(count, D))

            # Fill the pre-allocated arrays
            X[current_idx:current_idx+count] = samples
            y[current_idx:current_idx+count] = k

            current_idx += count

    # 3. Shuffle the result to mix components randomly
    # (Otherwise, output is sorted by component 0, then 1, etc.)
    perm = rng.permutation(n_samples)

    return X[perm], y[perm]

def gen_moons3(n_samples=800, noise=0.05, random_state=None):
    """Sample n_samples points by drawing class membership uniformly from {0, 1, 2}."""
    rng = np.random.default_rng(random_state)

    membership = rng.integers(0, 3, size=n_samples)
    counts = np.bincount(membership, minlength=3)
    n0, n1, n2 = counts.tolist()

    # Pass a tuple to generate exactly the needed number of points per class
    X01, y01 = make_moons(n_samples=(n0+n2, n1), noise=noise, random_state=random_state)
    y01 = y01.astype(int)
    idx0 = (y01 == 0)

    X0 = X01[idx0][:n0].copy()
    X1 = X01[y01 == 1][:n1].copy()
    X2 = X01[idx0][n0:n0+n2].copy()

    th = np.deg2rad(0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    X2 = X2 @ R.T + np.array([2.5, 0.5])

    y0 = np.zeros(len(X0), dtype=int)
    y1 = np.ones(len(X1), dtype=int)
    y2 = np.full(len(X2), 2, dtype=int)

    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.concatenate([y0, y1, y2])

    # Permute the dataset to mix the classes.
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X, y

class GammaMixture:
    def __init__(self, n_components=3, max_iter=1000, tol=1e-3, n_init=20, init_max_iter=15, random_state=None):
        """
        Parameters:
        -----------
        n_components : int
            Number of mixture components.
        max_iter : int
            Maximum iterations for the final deep exploitation run.
        tol : float
            Convergence tolerance.
        n_init : int
            Number of exploratory initializations (Stage 1).
        init_max_iter : int
            Number of EM iterations to run for each exploratory initialization.
        random_state : int, optional
            Controls the random seed.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_max_iter = init_max_iter
        self.random_state = random_state
        
        self.weights = None
        self.shapes = None  
        self.scales = None  
        self.converged_ = False
        self.lower_bound_ = -np.inf 

    def _estimate_log_prob(self, X, log_X, shapes, scales, weights):
        const_terms = np.sum(shapes * np.log(scales) + gammaln(shapes), axis=1)
        term_logx = np.dot(log_X, (shapes - 1.0).T)
        term_x = np.dot(X, (1.0 / scales).T)
        return (np.log(weights + 1e-10) + term_logx - term_x - const_terms)

    def _solve_shape(self, y, current_shape, n_iter=5):
        shape = current_shape.copy()
        for _ in range(n_iter):
            f_val = np.log(shape) - psi(shape) - y
            f_prime = 1.0 / shape - polygamma(1, shape)
            f_prime = np.minimum(f_prime, -1e-12)
            shape -= f_val / f_prime
            shape = np.clip(shape, 1e-3, 2000.0)
        return shape

    def _run_em(self, X, log_X, curr_shapes, curr_scales, curr_weights, max_iterations):
        """Core EM loop, abstracted so it can be run for short or long durations."""
        N, d = X.shape
        log_likelihood_history = []
        converged = False
        
        for it in range(max_iterations):
            # E-Step
            log_prob = self._estimate_log_prob(X, log_X, curr_shapes, curr_scales, curr_weights)
            log_prob_norm = logsumexp(log_prob, axis=1)
            
            if np.any(np.isnan(log_prob_norm)):
                log_likelihood_history.append(-np.inf)
                break

            current_ll = np.sum(log_prob_norm)
            log_likelihood_history.append(current_ll)
            
            resp = np.exp(log_prob - log_prob_norm[:, np.newaxis])

            # M-Step
            Nk = resp.sum(axis=0) + 1e-10
            curr_weights = Nk / N
            
            w_mean_x = np.dot(resp.T, X) / Nk[:, np.newaxis]
            w_mean_log_x = np.dot(resp.T, log_X) / Nk[:, np.newaxis]
            
            y = np.log(w_mean_x) - w_mean_log_x
            curr_shapes = self._solve_shape(y, curr_shapes)
            curr_scales = w_mean_x / curr_shapes
            
            # Convergence Check
            if it > 0 and abs(current_ll - log_likelihood_history[-2]) < self.tol:
                converged = True
                break
                
        final_ll = log_likelihood_history[-1] if log_likelihood_history else -np.inf
        return final_ll, curr_shapes, curr_scales, curr_weights, converged

    def fit(self, X):
        X = np.maximum(X, 1e-6) 
        N, d = X.shape
        log_X = np.log(X)
        
        rng = np.random.default_rng(self.random_state)
        
        # Base statistics for initialization
        global_mean = np.mean(X, axis=0)
        global_var = np.var(X, axis=0)
        base_shape = global_mean**2 / global_var
        base_scale = global_var / global_mean
        
        best_init_ll = -np.inf
        best_shapes = None
        best_scales = None
        best_weights = None

        # ==========================================
        # STAGE 1: Short Exploratory Runs
        # ==========================================
        for init_idx in range(self.n_init):
            # Generate random starting points
            curr_weights = np.ones(self.n_components) / self.n_components
            curr_shapes = np.tile(base_shape, (self.n_components, 1)) * rng.uniform(0.6, 1.4, (self.n_components, d))
            curr_scales = np.tile(base_scale, (self.n_components, 1)) * rng.uniform(0.6, 1.4, (self.n_components, d))

            # Run a short burst of EM
            ll, shapes, scales, weights, _ = self._run_em(
                X, log_X, curr_shapes, curr_scales, curr_weights, self.init_max_iter
            )
            
            # Keep track of the best basin of attraction
            if ll > best_init_ll:
                best_init_ll = ll
                best_shapes = shapes
                best_scales = scales
                best_weights = weights

        # ==========================================
        # STAGE 2: Deep Exploitation
        # ==========================================
        if best_shapes is None:
             raise ValueError("All initializations failed due to numerical instability.")
             
        final_ll, final_shapes, final_scales, final_weights, converged = self._run_em(
            X, log_X, best_shapes, best_scales, best_weights, self.max_iter
        )
        
        # Save final state
        self.lower_bound_ = final_ll
        self.shapes = final_shapes
        self.scales = final_scales
        self.weights = final_weights
        self.converged_ = converged
        
        if not self.converged_:
            warnings.warn(
                f"GammaMixture did not converge after {self.max_iter} iterations.",
                ConvergenceWarning
            )
            
        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted.")
        X = np.maximum(X, 1e-6)
        log_prob = self._estimate_log_prob(X, np.log(X), self.shapes, self.scales, self.weights)
        log_prob_norm = logsumexp(log_prob, axis=1)
        return np.exp(log_prob - log_prob_norm[:, np.newaxis])

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted.")
        X = np.maximum(X, 1e-6)
        log_prob = self._estimate_log_prob(X, np.log(X), self.shapes, self.scales, self.weights)
        return log_prob.argmax(axis=1)

class FCM:
    def __init__(self, **kargs):
        self.kwargs = kargs

    def fit(self, X):
        self.fcm_result = fuzz.cluster.cmeans(X.T, **self.kwargs)
        self.centers_ = self.fcm_result[0] # cluster centers (n_clusters, n_features)
        self.probs = self.fcm_result[1].T  # membership degrees (n_samples, n_clusters)
        self.labels_ = np.argmax(self.probs, axis=1) # hard cluster assignments (n_samples,)

def spectral_embedding(X, K, n_neighbors, mutual=True, size_threshold=1500):
    """
    Compute the spectral embedding of X using the graph Laplacian of the kNN graph.
    Parameters:
    - X: (n_samples, n_features) data matrix
    - K: int, number of dimensions for the spectral embedding
    - n_neighbors: int, number of neighbors for the kNN graph
    - mutual: bool, whether to use mutual kNN graph or not
    Returns:
    - X_embd: (n_samples, K) spectral embedding of X
    """
    
    G_dnn = kneighbors_graph(X, n_neighbors=n_neighbors) # directed graph
    G_mnn = G_dnn.multiply(G_dnn.T).toarray()

    if mutual==True:
        G = G_mnn
    else:
        G_nn = (G_dnn + G_dnn.T > 0).astype(int).toarray()
        G = G_nn

    L = np.diag(G.sum(axis=1)) - G
    n = L.shape[0]
    if n <= size_threshold:
        # full eigendecomposition for small datasets
        eigvals_all, eigvecs_all = eigh(L)
        eigvals = eigvals_all[:K]
        eigvecs = eigvecs_all[:, :K]
    else:
        # sparse eigendecomposition for large datasets
        n_eig = int(np.sqrt(L.shape[0]-1))  # number of eigenvalues to compute
        eigvals, eigvecs = eigsh(csr_matrix(L), k=n_eig, which='SM')
    X_embd = eigvecs[:, :K]

    return X_embd

def svd_embedding(X, K, size_threshold=1000, rsvd_seed=None):
    """
    Compute the SVD embedding of X, automatically routing to the most 
    efficient solver based on matrix size and sparsity.
    
    Parameters:
    - X: (n_samples, n_features) data matrix (dense or scipy.sparse).
    - K: int, number of dimensions for the embedding.
    - size_threshold: int, the dimension threshold to trigger randomized SVD.
    
    Returns:
    - X_embd: (n_samples, K) SVD embedding of X
    """
    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    
    if K > max_components:
        raise ValueError(f"K={K} cannot be greater than min(n_samples, n_features)={max_components}")

    # Condition 1: Is the matrix large enough to benefit from randomization?
    is_large = max_components >= size_threshold
    
    # Condition 2: Is K significantly smaller than the matrix dimensions?
    # (Randomized SVD loses its speed advantage if K is too large)
    is_k_small = K < (max_components / 10)

    # ---------------------------------------------------------
    # Path A: Large Matrix + Small K -> Randomized SVD
    # ---------------------------------------------------------
    if is_large and is_k_small:
        model = TruncatedSVD(n_components=K, algorithm='randomized', random_state=rsvd_seed)
        return model.fit_transform(X)
        
    # ---------------------------------------------------------
    # Path B: Small Matrix OR Large K -> Exact SVD
    # ---------------------------------------------------------
    else:
        if sp.issparse(X):
            # ARPACK is a deterministic, exact iterative solver optimized for sparse matrices
            model = TruncatedSVD(n_components=K, algorithm='arpack')
            return model.fit_transform(X)
        else:
            # For small dense matrices, LAPACK (standard exact SVD) is highly optimized and fastest.
            # full_matrices=False computes only the reduced/economy SVD, saving memory.
            U, Sigma, _ = svd(X, full_matrices=False)
            
            # Truncate to top K components
            U_k = U[:, :K]
            Sigma_k = Sigma[:K]
            
            # The embedding is U * Sigma
            return U_k * Sigma_k

class SpectralKM:
    def __init__(self, embedding='spectral', n_neighbors=5, mutual=True, rsvd_seed=None, **kwargs):
        self.embedding = embedding
        self.n_neighbors = n_neighbors
        self.mutual = mutual
        self.rsvd_seed = rsvd_seed
        self.clst = KMeans(**kwargs)
        self.K = kwargs['n_clusters']

        if embedding not in ['spectral', 'svd']:
            raise ValueError("Invalid embedding method. Choose from ['spectral', 'svd']")

    def fit(self, X):
        if self.embedding == 'spectral':
            self.X_embd = spectral_embedding(X, self.K, self.n_neighbors, self.mutual)
        elif self.embedding == 'svd':
            self.X_embd = svd_embedding(X, self.K, rsvd_seed=self.rsvd_seed)

        self.clst.fit(self.X_embd)
        self.labels_ = self.clst.labels_

class SpectralFCM:
    def __init__(self, embedding='spectral', n_neighbors=5, mutual=True, rsvd_seed=None, **kwargs):
        self.embedding = embedding
        self.n_neighbors = n_neighbors
        self.mutual = mutual
        self.rsvd_seed = rsvd_seed
        self.clst = FCM(**kwargs)
        self.K = kwargs['c']

        if embedding not in ['spectral', 'svd']:
            raise ValueError("Invalid embedding method. Choose from ['spectral', 'svd']")

    def fit(self, X):
        self.X = X
        if self.embedding == 'spectral':
            self.X_embd = spectral_embedding(X, self.K, self.n_neighbors, self.mutual)
        elif self.embedding == 'svd':
            self.X_embd = svd_embedding(X, self.K, rsvd_seed=self.rsvd_seed)

        self.clst.fit(self.X_embd)
        self.labels_ = self.clst.labels_
        self.probs = self.clst.probs


class SpectralGMM:
    def __init__(self, embedding='spectral', n_neighbors=5, mutual=True, rsvd_seed=None, **kwargs):
        self.embedding = embedding
        self.n_neighbors = n_neighbors
        self.mutual = mutual
        self.rsvd_seed = rsvd_seed
        self.clst = GaussianMixture(**kwargs)
        self.K = kwargs['n_components']

        if embedding not in ['spectral', 'svd']:
            raise ValueError("Invalid embedding method. Choose from ['spectral', 'svd']")

    def fit(self, X):
        self.X = X
        if self.embedding == 'spectral':
            self.X_embd = spectral_embedding(X, self.K, self.n_neighbors, self.mutual)
        elif self.embedding == 'svd':
            self.X_embd = svd_embedding(X, self.K, rsvd_seed=self.rsvd_seed)
        
        self.labels_ = self.clst.fit_predict(self.X_embd)
        self.probs = self.clst.predict_proba(self.X_embd)

class WeightedConformalClustering:
    def __init__(self, X_tr, X_cal, alpha=0.1):
        self.X_tr = X_tr
        self.X_cal = X_cal
        self.alpha = alpha

    def set_classifier(self, classifier, **kwargs):
        self.classifier = classifier
        if classifier == 'RF':
            self.clf = RandomForestClassifier(**kwargs)
        elif classifier == 'SVC':
            self.clf = SVC(**kwargs) # probability=True should be set in kwargs
            if not self.clf.probability:
                raise ValueError("For SVC, probability=True must be set")
        elif classifier == 'kNN':
            self.clf = KNeighborsClassifier(**kwargs)
        elif classifier in ['GMM', 'GaMM']:
            self.clf = None # use clustering probabilities directly
        else:
            raise ValueError("Invalid classification method. Choose from ['RF', 'SVC', 'kNN', 'GMM', 'GaMM']")

    def train(self, clustering, gen_seed_tr=0, **kwargs):
        if clustering == 'GMM':
            self.clst_tr = GaussianMixture(**kwargs)
            self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.K = kwargs['n_components']

        elif clustering == 'GaMM':
            self.clst_tr = GammaMixture(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.predict(self.X_tr)
            self.K = kwargs['n_components']

        elif clustering == 'KM':
            self.clst_tr = KMeans(**kwargs)
            self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.K = kwargs['n_clusters']

        elif clustering == 'FCM':
            self.clst_tr = FCM(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.labels_
            self.K = kwargs['c']

        elif clustering == 'Hier':
            self.clst_tr = AgglomerativeClustering(**kwargs)
            self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.K = kwargs['n_clusters']

        elif clustering == 'SpectralKM':
            self.clst_tr = SpectralKM(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.labels_
            self.K = kwargs['n_clusters']

        elif clustering == 'SpectralFCM':
            self.clst_tr = SpectralFCM(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.y_tr = self.clst_tr.labels_
            self.K = kwargs['c']

        # Stochastic clustering methods (seeds for sampling)
        elif clustering == 'GMMS':
            self.clst_tr = GaussianMixture(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.clst_tr_prob = self.clst_tr.predict_proba(self.X_tr) # (n_tr, K)
            self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.K = kwargs['n_components']

        elif clustering == 'GaMMS':
            self.clst_tr = GammaMixture(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.clst_tr_prob = self.clst_tr.predict_proba(self.X_tr) # (n_tr, K)
            self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.K = kwargs['n_components']

        elif clustering == 'FCMS':
            self.clst_tr = FCM(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.clst_tr_prob = self.clst_tr.probs # (n_tr, K)
            self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.K = kwargs['c']

        elif clustering == 'SpectralGMMS':
            self.clst_tr = SpectralGMM(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.clst_tr_prob = self.clst_tr.probs # (n_tr, K)
            self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.K = kwargs['n_components']

        elif clustering == 'SpectralFCMS':
            self.clst_tr = SpectralFCM(**kwargs)
            self.clst_tr.fit(self.X_tr)
            self.clst_tr_prob = self.clst_tr.probs # (n_tr, K)
            self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.K = kwargs['c']

        else:
            raise ValueError("Invalid clustering method. Choose from ['GMM', 'GMMS', 'GaMM', 'GaMMS', 'KM', 'FCM', 'FCMS', 'Hier', 'SpectralGMMS', 'SpectralKM', 'SpectralFCM', 'SpectralFCMS']")
        
        # Fit the classifier
        if self.classifier in ['RF', 'SVC', 'kNN']:
            self.clf.fit(self.X_tr, self.y_tr)
            if self.clf.classes_.shape[0] != self.K:
                raise ValueError("Number of classes in classifier does not match number of clusters")
        elif self.classifier == 'GMM' and clustering in ['GMM', 'GMMS']:
            self.clf = self.clst_tr
        elif self.classifier == 'GaMM' and clustering in ['GaMM', 'GaMMS']:
            self.clf = self.clst_tr
        else:
            raise ValueError("Classifier and clustering method are not compatible. For GMM classifier, use GMM clustering. For GaMM classifier, use GaMM clustering.")
        
    def conformal_set(self, x, clustering, gen_seed_cal=0, method='narc', clip_min=1e-6, **kwargs):
        """
        Returns:
        - prediction_set: array of shape (1, K) with row = prediction
        """
        # Augment the calibration set with the new data point x 
        if x.ndim == 1:
            raise ValueError("Input x must be a 2D array with shape (1, n_features)")
        
        n_cal = len(self.X_cal)
        self.X_aug = np.vstack([self.X_cal, x]) # (n_cal+1, n_features)
        
        if clustering == 'GMM':
            self.clst_aug = GaussianMixture(**kwargs)
            self.y_aug = self.clst_aug.fit_predict(self.X_aug)
            self.y_cal = self.y_aug[:-1] # only get the cluster label for the original calibration set
            self.y_test = self.y_aug[-1] # get the cluster label for the test point
            self.clst_cal_weight = np.ones(n_cal)
            self.clst_test_weight = np.zeros(self.K)
            self.clst_test_weight[self.y_test] = 1.0 # weight for the test point based on the cluster label

        elif clustering == 'GaMM':
            self.clst_aug = GammaMixture(**kwargs)
            self.clst_aug.fit(self.X_aug)
            self.y_aug = self.clst_aug.predict(self.X_aug)
            self.y_cal = self.y_aug[:-1] # only get the cluster label for the original calibration set
            self.y_test = self.y_aug[-1] # get the cluster label for the test point
            self.clst_cal_weight = np.ones(n_cal)
            self.clst_test_weight = np.zeros(self.K)
            self.clst_test_weight[self.y_test] = 1.0 # weight for the test point based on the cluster label

        elif clustering == 'KM':
            self.clst_aug = KMeans(**kwargs)
            self.y_aug = self.clst_aug.fit_predict(self.X_aug)
            self.y_cal = self.y_aug[:-1] # only get the cluster label for the original calibration set
            self.y_test = self.y_aug[-1] # get the cluster label for the test point
            self.clst_cal_weight = np.ones(n_cal)
            self.clst_test_weight = np.zeros(self.K)
            self.clst_test_weight[self.y_test] = 1.0 # weight for the test point based on the cluster label

        elif clustering == 'FCM':
            self.clst_aug = FCM(**kwargs)
            self.clst_aug.fit(self.X_aug)
            self.y_cal = self.clst_aug.labels_[:-1] # only get the cluster label for the original calibration set
            self.y_test = self.clst_aug.labels_[-1] # get the cluster label for the test point
            self.clst_cal_weight = np.ones(n_cal)            
            self.clst_test_weight = np.zeros(self.K)
            self.clst_test_weight[self.y_test] = 1.0 # weight for the test point based on the cluster label

        elif clustering == 'Hier':
            self.clst_aug = AgglomerativeClustering(**kwargs)
            self.y_aug = self.clst_aug.fit_predict(self.X_aug)
            self.y_cal = self.y_aug[:-1] # only get the cluster label for the original calibration set
            self.y_test = self.y_aug[-1] # get the cluster label for the test point
            self.clst_cal_weight = np.ones(n_cal)
            self.clst_test_weight = np.zeros(self.K)
            self.clst_test_weight[self.y_test] = 1.0 # weight for the test point based on the cluster label

        elif clustering == 'SpectralKM':
            self.clst_aug = SpectralKM(**kwargs)
            self.clst_aug.fit(self.X_aug)
            self.y_cal = self.clst_aug.labels_[:-1] # only get the cluster label for the original calibration set
            self.y_test = self.clst_aug.labels_[-1] # get the cluster label for the test point
            self.clst_cal_weight = np.ones(n_cal)
            self.clst_test_weight = np.zeros(self.K)
            self.clst_test_weight[self.y_test] = 1.0 # weight for the test point based on the cluster label

        elif clustering == 'SpectralFCM':
            self.clst_aug = SpectralFCM(**kwargs)
            self.clst_aug.fit(self.X_aug)
            self.y_cal = self.clst_aug.labels_[:-1] # only get the cluster label for the original calibration set
            self.y_test = self.clst_aug.labels_[-1] # get the cluster label for the test point
            self.clst_cal_weight = np.ones(n_cal)
            self.clst_test_weight = np.zeros(self.K)
            self.clst_test_weight[self.y_test] = 1.0 # weight for the test point based on the cluster label

        # Stochastic clustering methods (seeds for sampling)
        elif clustering == 'GMMS':
            self.clst_aug = GaussianMixture(**kwargs)
            self.clst_aug.fit(self.X_aug)
            clst_aug_prob = self.clst_aug.predict_proba(self.X_aug) # (n_cal+1, K)
            self.clst_cal_prob = clst_aug_prob[:n_cal, :] # (n_cal, K) probabilities for the original calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) stochastic cluster labels for the original calibration set
            self.clst_cal_weight = self.clst_cal_prob[np.arange(n_cal), self.y_cal] # (n_cal,) weights for the original calibration set based on the clustering probabilities
            self.clst_test_weight = clst_aug_prob[-1, :] # (K,) weights for the test point based on the clustering probabilities

        elif clustering == 'GaMMS':
            self.clst_aug = GammaMixture(**kwargs)
            self.clst_aug.fit(self.X_aug)
            clst_aug_prob = self.clst_aug.predict_proba(self.X_aug) # (n_cal+1, K)
            self.clst_cal_prob = clst_aug_prob[:n_cal, :] # (n_cal, K) probabilities for the original calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) stochastic cluster labels for the original calibration set
            self.clst_cal_weight = self.clst_cal_prob[np.arange(n_cal), self.y_cal] # (n_cal,) weights for the original calibration set based on the clustering probabilities
            self.clst_test_weight = clst_aug_prob[-1, :] # (K,) weights for the test point based on the clustering probabilities

        elif clustering == 'FCMS':
            self.clst_aug = FCM(**kwargs)
            self.clst_aug.fit(self.X_aug)
            clst_aug_prob = self.clst_aug.probs # (n_cal+1, K)
            self.clst_cal_prob = clst_aug_prob[:n_cal, :] # (n_cal, K) probabilities for the original calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) stochastic cluster labels for the original calibration set
            self.clst_cal_weight = self.clst_cal_prob[np.arange(n_cal), self.y_cal] # (n_cal,) weights for the original calibration set based on the clustering probabilities
            self.clst_test_weight = clst_aug_prob[-1, :] # (K,) weights for the test point based on the clustering probabilities

        elif clustering == 'SpectralGMMS':
            self.clst_aug = SpectralGMM(**kwargs)
            self.clst_aug.fit(self.X_aug)
            clst_aug_prob = self.clst_aug.probs # (n_cal+1, K)
            self.clst_cal_prob = clst_aug_prob[:n_cal, :] # (n_cal, K) probabilities for the original calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) stochastic cluster labels for the original calibration set
            self.clst_cal_weight = self.clst_cal_prob[np.arange(n_cal), self.y_cal] # (n_cal,) weights for the original calibration set based on the clustering probabilities
            self.clst_test_weight = clst_aug_prob[-1, :] # (K,) weights for the test point based on the clustering probabilities set

        elif clustering == 'SpectralFCMS':
            self.clst_aug = SpectralFCM(**kwargs)
            self.clst_aug.fit(self.X_aug)
            clst_aug_prob = self.clst_aug.probs # (n_cal+1, K)
            self.clst_cal_prob = clst_aug_prob[:n_cal, :] # (n_cal, K) probabilities for the original calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) stochastic cluster labels for the original calibration set
            self.clst_cal_weight = self.clst_cal_prob[np.arange(n_cal), self.y_cal] # (n_cal,) weights for the original calibration set based on the clustering probabilities
            self.clst_test_weight = clst_aug_prob[-1, :] # (K,) weights for the test point based on the clustering probabilities set

        else:
            raise ValueError("Invalid clustering method. Choose from ['GMM', 'GMMS', 'GaMM', 'GaMMS', 'KM', 'FCM', 'FCMS', 'Hier', 'SpectralGMMS', 'SpectralKM', 'SpectralFCM', 'SpectralFCMS']")
        
        # Get calibration probabilities
        self.cal_prob = self.clf.predict_proba(self.X_cal) # (n_cal, K)
        self.test_prob = self.clf.predict_proba(x) # (1, K)
        
        # # Align calibration labels
        # y_cal_hat = self.clf.predict(self.X_cal)
        # align_to_tr = label_alignment(y_cal_hat, self.y_cal, self.K) # align the obtained calibration labels to the training labels

        # Align calibration labels using the augmented data (calibration data + x)
        y_aug_hat = self.clf.predict(self.X_aug) # (n_cal+1,)
        if clustering in ['GMMS', 'GaMMS', 'FCMS', 'SpectralGMMS', 'SpectralFCMS']:
            y_cal_clst = np.argmax(self.clst_cal_prob, axis=1) # most probable clusters of the calibration data (instead of the sampled labels)
        else:
            y_cal_clst = self.y_cal # hard clustering labels of the calibration data
        y_test_clst = np.argmax(self.clst_test_weight) # cluster label of x (one-hot weight for hard clustering; most probable cluster for stochastic clustering)
        y_aug_clst = np.append(y_cal_clst, y_test_clst) # (n_cal+1,) cluster labels of the augmented data
        align_to_tr = label_alignment(y_aug_hat, y_aug_clst, self.K) # align the obtained augmented labels to the training labels
        self.y_cal_al = align_to_tr[self.y_cal] # aligned calibration labels
    
        # Compute the conformal set for the new data point x
        if method == 'arc':
            _conformal = conformal_arc
        elif method == 'arc_ne':
            _conformal = conformal_arc_ne
        elif method == 'narc':
            _conformal = conformal_narc
        elif method == 'clp':
            _conformal = conformal_clp
        elif method == 'clp_ne':
            _conformal = conformal_clp_ne
        else:
            raise ValueError("Invalid method. Choose from ['arc', 'arc_ne', 'narc', 'clp', 'clp_ne']")
        
        # Calculate the weights
        weight_cal_numerator = self.cal_prob[np.arange(n_cal), self.y_cal_al] # use classifier probabilities for numerator
        weight_test_numerator = self.test_prob # use classifier probabilities for numerator (no alignment for test point)

        # Clip the denominators
        self.clst_cal_weight_clipped = np.maximum(self.clst_cal_weight, clip_min)
        self.clst_test_weight_al = self.clst_test_weight[np.argsort(align_to_tr)] # align based on align_to_tr^{-1}
        self.clst_test_weight_clipped = np.maximum(self.clst_test_weight_al, clip_min)
        weight_cal_raw = weight_cal_numerator / self.clst_cal_weight_clipped # (n_cal,) raw weights for the calibration set
        #weight_test_raw = weight_test_numerator / self.clst_test_weight_clipped # (1, K)
        weight_test_raw = weight_test_numerator / self.clst_test_weight_clipped # (1, K) test weights

        # Identify cases where both the numerator and denominator are very small
        cal_mask = (weight_cal_numerator < clip_min) & (self.clst_cal_weight < clip_min)
        test_mask = (weight_test_numerator < clip_min) & (self.clst_test_weight_al < clip_min)

        weight_cal_raw[cal_mask] = 1.0 # set raw weights to 1 for calibration points where both numerator and denominator are very small
        weight_test_raw[test_mask] = 1.0 # set raw weights to 1 for test points where both numerator and denominator are very small

        self.weight = scale_weights(weight_cal_raw, weight_test_raw) # (n_cal, 1, K) array of scaled weights

        conformal_set_bool = _conformal(self.cal_prob, self.y_cal_al, self.test_prob, alpha=self.alpha, weight=self.weight) # (1, K) boolean array for the conformal set
        
        # Convert boolean to indices of the conformal set
        return np.where(conformal_set_bool[0])[0]  # array of indices (K,) for the conformal set
    
    def validate(self, y, conformal_set, X_al, y_al):
        """
        Parameters:
        - y: true label of the test point (scalar)
        - conformal_set: array of indices in the conformal set (K,)
        - X_al: alignment data (n_al, p)
        - y_al: true labels of the alignment data (n_al,)
        Returns:
        """

        # Oracle alignment using the alignment data
        align_true_to_conformal = label_alignment(self.clf.predict(X_al), y_al, self.K)
        y_aligned = align_true_to_conformal[y]
        coverage = int(y_aligned in conformal_set) # 1 if the true label is in the conformal set, 0 otherwise
        sizes = len(conformal_set) # size of the conformal set

        return coverage, sizes

class ConformalClustering:
    def __init__(self, X_tr, X_cal, alpha=0.1):
        self.X_tr = X_tr
        self.X_cal = X_cal
        self.alpha = alpha

    def set_classifier(self, classifier, clf=None, **kwargs):
        self.classifier = classifier
        if classifier == 'RF':
            self.clf = RandomForestClassifier(**kwargs)
        elif classifier == 'SVC':
            self.clf = SVC(**kwargs) # probability=True should be set in kwargs
            if not self.clf.probability:
                raise ValueError("For SVC, probability=True must be set")
        elif classifier == 'kNN':
            self.clf = KNeighborsClassifier(**kwargs)
        elif classifier in ['GMM', 'GaMM']:
            self.clf = None # use clustering probabilities directly
        elif classifier == 'pretrain':
            if clf is None:
                raise ValueError("For 'pretrain' classifier, clf must be provided")
            elif not isinstance(clf, (RandomForestClassifier, SVC, KNeighborsClassifier, GaussianMixture, GammaMixture)):
                raise ValueError("Pre-trained clf must be an instance of RandomForestClassifier, SVC, KNeighborsClassifier, GaussianMixture, or GammaMixture")
            else:
                self.clf = clf
        else:
            raise ValueError("Invalid classification method. Choose from ['RF', 'SVC', 'kNN', 'GMM', 'GaMM', 'pretrain']")
        
    def fit(self, clustering, gen_seed_tr=0, gen_seed_cal=0, **kwargs):
        if clustering == 'GMM':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = GaussianMixture(**kwargs)
                self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = GaussianMixture(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['n_components']

        elif clustering == 'GaMM':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = GammaMixture(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.y_tr = self.clst_tr.predict(self.X_tr)
            self.clst_cal = GammaMixture(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.predict(self.X_cal)
            self.K = kwargs['n_components']

        elif clustering == 'KM':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = KMeans(**kwargs)
                self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = KMeans(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['n_clusters']

        elif clustering == 'FCM':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = FCM(**kwargs)
                self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = FCM(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['c']

        elif clustering == 'Hier':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = AgglomerativeClustering(**kwargs)
                self.y_tr = self.clst_tr.fit_predict(self.X_tr)
            self.clst_cal = AgglomerativeClustering(**kwargs)
            self.y_cal = self.clst_cal.fit_predict(self.X_cal)
            self.K = kwargs['n_clusters']

        elif clustering == 'SpectralKM':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = SpectralKM(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.y_tr = self.clst_tr.labels_
            self.clst_cal = SpectralKM(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.labels_
            self.K = kwargs['n_clusters']

        elif clustering == 'SpectralFCM':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = SpectralFCM(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.y_tr = self.clst_tr.labels_
            self.clst_cal = SpectralFCM(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.y_cal = self.clst_cal.labels_
            self.K = kwargs['c']

        # Stochastic clustering methods (seeds for sampling)
        elif clustering == 'GMMS':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = GaussianMixture(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.clst_tr_prob = self.clst_tr.predict_proba(self.X_tr) # (n_tr, K)
                self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.clst_cal = GaussianMixture(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.clst_cal_prob = self.clst_cal.predict_proba(self.X_cal) # (n_cal, K) probabilities for the calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) sampled cluster labels for the calibration set
            self.K = kwargs['n_components']

        elif clustering == 'GaMMS':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = GammaMixture(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.clst_tr_prob = self.clst_tr.predict_proba(self.X_tr) # (n_tr, K)
                self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.clst_cal = GammaMixture(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.clst_cal_prob = self.clst_cal.predict_proba(self.X_cal) # (n_cal, K) probabilities for the calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) sampled cluster labels for the calibration set
            self.K = kwargs['n_components']

        elif clustering == 'FCMS':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = FCM(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.clst_tr_prob = self.clst_tr.probs # (n_tr, K)
                self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.clst_cal = FCM(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.clst_cal_prob = self.clst_cal.probs # (n_cal, K) probabilities for the calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) sampled cluster labels for the calibration set
            self.K = kwargs['c']

        elif clustering == 'SpectralGMMS':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = SpectralGMM(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.clst_tr_prob = self.clst_tr.probs # (n_tr, K)
                self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.clst_cal = SpectralGMM(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.clst_cal_prob = self.clst_cal.probs # (n_cal, K) probabilities for the calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) sampled cluster labels for the calibration set
            self.K = kwargs['n_components']

        elif clustering == 'SpectralFCMS':
            if self.classifier == 'pretrain':
                pass # skip clustering if using pre-trained classifier
            else:
                self.clst_tr = SpectralFCM(**kwargs)
                self.clst_tr.fit(self.X_tr)
                self.clst_tr_prob = self.clst_tr.probs # (n_tr, K)
                self.y_tr = soft_predict(self.clst_tr_prob, seed=gen_seed_tr) # (n_tr,) sampled cluster labels for the training set
            self.clst_cal = SpectralFCM(**kwargs)
            self.clst_cal.fit(self.X_cal)
            self.clst_cal_prob = self.clst_cal.probs # (n_cal, K) probabilities for the calibration set
            self.y_cal = soft_predict(self.clst_cal_prob, seed=gen_seed_cal) # (n_cal,) sampled cluster labels for the calibration set
            self.K = kwargs['c']

        else:
            raise ValueError("Invalid clustering method. Choose from ['GMM', 'GMMS', 'GaMM', 'GaMMS', 'KM', 'FCM', 'FCMS', 'Hier', 'SpectralGMMS', 'SpectralKM', 'SpectralFCM', 'SpectralFCMS']")
        
        # Fit the classifier
        if self.classifier == 'pretrain':
            pass
        elif self.classifier in ['RF', 'SVC', 'kNN']:
            self.clf.fit(self.X_tr, self.y_tr)
            if self.clf.classes_.shape[0] != self.K:
                raise ValueError("Number of classes in classifier does not match number of clusters")
        elif self.classifier == 'GMM' and clustering in ['GMM', 'GMMS', 'GMM-Gen']:
            self.clf = self.clst_tr
        elif self.classifier == 'GaMM' and clustering in ['GaMM', 'GaMMS']:
            self.clf = self.clst_tr
        else:
            raise ValueError("Classifier and clustering method are not compatible. For GMM classifier, use GMM clustering. For GaMM classifier, use GaMM clustering.")
        
        # Get calibration probabilities
        self.cal_prob = self.clf.predict_proba(self.X_cal) # (n_cal, K)
        
        # Align calibration labels
        y_cal_hat = self.clf.predict(self.X_cal)
        align_to_tr = label_alignment(y_cal_hat, self.y_cal, self.K)
        self.y_cal_al = align_to_tr[self.y_cal] # aligned calibration labels

    def conformal_set(self, X_test, method='narc'):
        """
        Parameters:
        - X_test: test points (n_test, K)
        Returns:
        - prediction_set: array of shape (n_test, K) with each row = prediction (boolean vector)
        """
        if method == 'arc':
            _conformal = conformal_arc
        elif method == 'arc_ne':
            _conformal = conformal_arc_ne
        elif method == 'narc':
            _conformal = conformal_narc
        elif method == 'clp':
            _conformal = conformal_clp
        elif method == 'clp_ne':
            _conformal = conformal_clp_ne
        else:
            raise ValueError("Invalid method. Choose from ['arc', 'arc_ne', 'narc', 'clp', 'clp_ne']")


        test_prob = self.clf.predict_proba(X_test)

        return _conformal(self.cal_prob, self.y_cal_al, test_prob, alpha=self.alpha)
            
    def validate(self, X_val, y_val, X_al, y_al, method='narc'):
        """
        Parameters:
        - X_val: validation data (n_val, p)
        - y_val: true labels of the validation data (n_val,)
        - X_al: alignment data (n_al, p)
        - y_al: true labels of the alignment data (n_al,)
        Returns:
        """

        if method not in ['arc', 'arc_ne', 'narc', 'clp', 'clp_ne']:
            raise ValueError("Invalid method. Choose from ['arc', 'arc_ne', 'narc', 'clp', 'clp_ne']")
        
        # Oracle alignment using the alignment data
        C_bool = self.conformal_set(X_al, method=method) # conformal sets as boolean (n_al, K)
        C = csr_array(C_bool.astype(int)) # convert to sparse matrix

        n_al = len(y_al)
        y_mat = csr_array(  
            (np.ones(n_al), (np.arange(n_al), y_al)),
            shape=(n_al, self.K)
        ) # labels to one-hot encoding (row = e_{y_i})

        Q = -y_mat.T @ C # (K, K) cost matrix
        _, align_true_to_conformal = linear_sum_assignment(Q.toarray()) 

        # Compute conformal sets for the validation data
        conformal_sets_bool = self.conformal_set(X_val, method=method) # (n_val, K)
        conformal_sets = [np.where(row)[0] for row in conformal_sets_bool]  # list of arrays
        sizes = [len(s) for s in conformal_sets]

        # Align true labels to conformal sets
        y_val_aligned = align_true_to_conformal[y_val]
        n_val = len(y_val)

        # Check coverage and set sizes
        if n_val == 1:
            coverage = int(y_val_aligned[0] in conformal_sets[0])
            return coverage, sizes[0]
        else:
            coverage = np.array([int(y_val_aligned[i] in conformal_sets[i]) for i in range(n_val)])
            return coverage, sizes

class GMMCutoff:
    def __init__(self, X, alpha=0.1):
        self.X = X
        self.alpha = alpha

    def fit(self, **kwargs):
        self.gmm = GaussianMixture(**kwargs)
        self.y_clst = self.gmm.fit_predict(self.X)
        self.K = kwargs['n_components']

    def cutoff_set(self, X_test):
        test_prob = self.gmm.predict_proba(X_test)
        return cutoff(test_prob, self.alpha)
    
    def validate(self, X_val, y_val, X_al, y_al):
        # Oracle alignment using the alignment data
        C_bool = self.cutoff_set(X_al) # conformal sets as boolean (n_al, K)
        C = csr_array(C_bool.astype(int)) # convert to sparse matrix

        n_al = len(y_al)
        y_mat = csr_array(  
            (np.ones(n_al), (np.arange(n_al), y_al)),
            shape=(n_al, self.gmm.n_components)
        ) # labels to one-hot encoding (row = e_{y_i})

        Q = -y_mat.T @ C # (K, K) cost matrix
        _, align_true_to_cutoff = linear_sum_assignment(Q.toarray()) 

        # Compute cutoff sets for the validation data
        cutoff_sets_bool = self.cutoff_set(X_val) # (n_val, K)
        cutoff_sets = [np.where(row)[0] for row in cutoff_sets_bool]  # list of arrays
        sizes = [len(s) for s in cutoff_sets]

        # Align true labels to cutoff sets
        y_val_aligned = align_true_to_cutoff[y_val]
        n_val = len(y_val)

        # Check coverage and set sizes
        if n_val == 1:
            coverage = int(y_val_aligned[0] in cutoff_sets[0])
            return coverage, sizes[0]
        else:
            coverage = np.array([int(y_val_aligned[i] in cutoff_sets[i]) for i in range(n_val)])
            return coverage, sizes
    
class GaMMCutoff:
    def __init__(self, X, alpha=0.1):
        self.X = X
        self.alpha = alpha

    def fit(self, **kwargs):
        self.gamm = GammaMixture(**kwargs)
        self.gamm.fit(self.X)
        self.y_clst = self.gamm.predict(self.X)
        self.K = kwargs['n_components']

    def cutoff_set(self, X_test):
        test_prob = self.gamm.predict_proba(X_test)
        return cutoff(test_prob, self.alpha)
    
    def validate(self, X_val, y_val, X_al, y_al):
        # Oracle alignment using the alignment data
        C_bool = self.cutoff_set(X_al) # conformal sets as boolean (n_al, K)
        C = csr_array(C_bool.astype(int)) # convert to sparse matrix

        n_al = len(y_al)
        y_mat = csr_array(  
            (np.ones(n_al), (np.arange(n_al), y_al)),
            shape=(n_al, self.gamm.n_components)
        ) # labels to one-hot encoding (row = e_{y_i})

        Q = -y_mat.T @ C # (K, K) cost matrix
        _, align_true_to_cutoff = linear_sum_assignment(Q.toarray()) 

        # Compute cutoff sets for the validation data
        cutoff_sets_bool = self.cutoff_set(X_val) # (n_val, K)
        cutoff_sets = [np.where(row)[0] for row in cutoff_sets_bool]  # list of arrays
        sizes = [len(s) for s in cutoff_sets]

        # Align true labels to cutoff sets
        y_val_aligned = align_true_to_cutoff[y_val]
        n_val = len(y_val)

        # Check coverage and set sizes
        if n_val == 1:
            coverage = int(y_val_aligned[0] in cutoff_sets[0])
            return coverage, sizes[0]
        else:
            coverage = np.array([int(y_val_aligned[i] in cutoff_sets[i]) for i in range(n_val)])
            return coverage, sizes

def cutoff(test_prob, alpha=0.1):
    sorted_ids = np.argsort(-test_prob, axis=1)  # descending order
    sorted_probs = np.take_along_axis(test_prob, sorted_ids, axis=1)
    test_prob_rank = np.argsort(sorted_ids, axis=1)
    sorted_probs_cumsum = np.cumsum(sorted_probs, axis=1)
    cutoff = sorted_probs_cumsum < 1-alpha
    cutoff[np.arange(len(test_prob)),np.argmin(cutoff, axis=1)] = True
    prediction_set = np.take_along_axis(cutoff, test_prob_rank, axis=1)
    return prediction_set

def gmm_response_prob(X, means, covariances, weights):
    """
    Computes the response probability vector (responsibilities) for data X 
    under a Gaussian Mixture Model.

    Parameters:
    - X: (n_samples, D) array of data points.
    - means: (K, D) array of component means.
    - covariances: (K, D, D) array of component covariance matrices.
    - weights: (K,) array of component mixing proportions (must sum to 1).

    Returns:
    - responsibilities: (n_samples, K) array where the (i, k)-th element is the 
      probability that the i-th sample belongs to the k-th mixture component.
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    weights = np.asarray(weights)
    
    n_samples, D = X.shape
    K = len(weights)
    
    # Pre-allocate array for the weighted log-probabilities: 
    # log(w_k) + log(N(x | mu_k, Sigma_k))
    weighted_log_probs = np.empty((n_samples, K))
    
    # Compute the log-probability for each component
    for k in range(K):
        # multivariate_normal.logpdf is highly optimized and vectorized over X
        log_pdf = multivariate_normal.logpdf(X, mean=means[k], cov=covariances[k])
        
        # Add log weight (use a tiny epsilon to prevent log(0) if a weight is exactly 0)
        weighted_log_probs[:, k] = np.log(weights[k] + 1e-15) + log_pdf
        
    # Apply the Log-Sum-Exp trick to compute the log-marginal probabilities
    # This acts as the stable denominator for normalization
    log_prob_norm = logsumexp(weighted_log_probs, axis=1, keepdims=True)
    
    # Compute log-responsibilities by subtracting the log-denominator
    log_responsibilities = weighted_log_probs - log_prob_norm
    
    # Exponentiate to return back to standard probability space [0, 1]
    return np.exp(log_responsibilities)

def scale_weights(weight_raw, weight_other):
    """
    Parameters:
    - weight_raw: (n,) array of raw weights for the calibration set
    - weight_other: (m, K) array of raw weights for m test points and K classes

    Returns:
    - scaled_weights: (n, m, K) array of scaled weights for the calibration set, where each weight is divided by the sum of all weights (calibration + test) for the corresponding class: weight_ijk = weight_raw_i / (sum(weight_raw) + weight_other_jk) for i in [1,...,n], j in [1,...,m], k in [1,...,K]
    """
    # 1. Precompute the sum of weight_raw (a scalar)
    total_weight = np.sum(weight_raw)
    
    # 2. Expand dimensions and divide using broadcasting
    # weight_raw[:, np.newaxis, np.newaxis] shapes it to (n, 1, 1)
    # (total_weight + weight_other) has shape (m, K)
    result = weight_raw[:, np.newaxis, np.newaxis] / (total_weight + weight_other)
    
    return result

def soft_predict(probs, seed=0):
    n_samples, K = probs.shape
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
        
    cum_probs = probs.cumsum(axis=1)
    cum_probs[:, -1] = 1.0 # force the last column to be exactly 1.0 to prevent numerical issues
    u = rng.random((probs.shape[0], 1)) # draw one uniform random number per row
    y = (u < cum_probs).argmax(axis=1) # smallest index where u < cum_probs = find wher u falls in the cumulative distribution

    return y
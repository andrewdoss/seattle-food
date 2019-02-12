import numpy as np
from sklearn.cluster import KMeans

class GapAnalysis(object):
    """Performs gap analysis for evaluating clustering results.

    This object implements the gap analysis method from Elements
    of Statistical Learning, Hastie Et. Al.

    Paramaters
    ----------
    cluster_method: sklearn KMeans class (arg b/c might add EM object support)
     KMeans object.
    max_clusters: int, optional
     Maximum number of clusters to consider. Default is 10.
    num_samples: int, optional
     Number of randomly sampled datasets to use for computing wk_null.
    random_state: int, optional
     Random seed to use.
    cov_type: string, optional
     Covariance type for a GaussianMixture model. Also indicates model type.

    Attributes
    ----------
    wk: 1d-array
     Within clusters inertia values on original dataset.
    wk_null: 1d-array
     Mean within clusters inertia values on null dataset samples.
    wk_null_stdev: 1d-array
     Stdev of within clusters inertia values on null dataset samples.
    """

    def __init__(self, cluster_method, max_clusters=10, num_samples=20,
                 random_state=0, cov_type=None):
        self.cluster_method = cluster_method
        self.max_clusters = max_clusters
        self.num_samples = num_samples
        self.random_state = random_state
        self.cov_type = cov_type

    def fit(self, X):
        """"Fit clustering_method to original data and to noise samples."""
        rgen = np.random.RandomState(self.random_state)
        self.wk_ = np.zeros(self.max_clusters)
        self.wk_null_ = np.zeros(self.max_clusters)
        self.wk_null_stdev_ = np.zeros(self.max_clusters)
        self.gap_ = np.zeros(self.max_clusters)
        self.gap_stdev_ = np.zeros(self.max_clusters)

        # Generate results for original dataset
        for idx, n in enumerate(range(1,self.max_clusters+1)):
            if self.cov_type is not None:
                cm = self.cluster_method(n_components=n,
                                         covariance_type=self.cov_type)
                cm.fit(X)
                self.wk_[idx] = cm.bic(X)
            else:
                cm = self.cluster_method(n_clusters=n)
                cm.fit(X)
                self.wk_[idx] = np.log(cm.inertia_)


        # Generate results for noise samples
        for idx, n in enumerate(range(1,self.max_clusters+1)):
            temp_wk = np.zeros(self.num_samples)
            for i in range(self.num_samples):
                if self.cov_type is not None:
                    cm = self.cluster_method(n_components=n,
                                             covariance_type=self.cov_type)
                    X_sample = self.generate_sample(X, rgen)
                    cm.fit(X_sample)
                    temp_wk[i] = cm.bic(X_sample)
                else:
                    cm = self.cluster_method(n_clusters=n)
                    cm.fit(self.generate_sample(X, rgen))
                    temp_wk[i] = cm.inertia_


            if self.cov_type is not None:
                self.wk_null_[idx] = np.mean(temp_wk)
                self.wk_null_stdev_[idx] = np.std(temp_wk)
                temp_gap = temp_wk - self.wk_[idx]
            else:
                self.wk_null_[idx] = np.mean(np.log(temp_wk))
                self.wk_null_stdev_[idx] = np.std(np.log(temp_wk))
                temp_gap = np.log(temp_wk) - self.wk_[idx]
            self.gap_[idx] = np.mean(temp_gap)
            self.gap_stdev_[idx] = (np.std(temp_gap) *
                                    np.sqrt(1+(1/self.num_samples)))

    def generate_sample(self, X, rgen):
        """Generate random sample for clustering."""
        sample_X = np.zeros(X.shape)
        for j in range(X.shape[1]):
            min_j = np.min(X[:,j])
            max_j = np.max(X[:,j])
            sample_X[:,j] = rgen.uniform(min_j, max_j, size=X.shape[0])
        return sample_X

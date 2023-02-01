import numpy as np
import sklearn 
import sklearn.decomposition
import time
class pca_with_missing_values(sklearn.decomposition.PCA):
    def __init__(self, **kargs):
        if 'n_outer_loops' in kargs:
            self.n_outer_loops_ = kargs['n_outer_loops']
            del kargs['n_outer_loops']
        else:
            self.n_outer_loops_ = 1
        if 'save_space' in kargs:
            self.save_space = kargs['save_space']
            del kargs['save_space'] 
        else:
            self.save_space = False
        self.time = None # float, default None, store time that takes to run fit_transform
        super(pca_with_missing_values, self).__init__(**kargs)
    def fit_transform(self, X, y = None):
        """Learn a PCA model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        W : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.
        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data.
        """
        start_time = time.time()
        X = np.maximum(X, 0) # first impute missing values (negative numbers) to zero. 
        for iter in range(self.n_outer_loops_):
            # if the initialization is given, set self.init to custom
            W = super(pca_with_missing_values, self).fit_transform(X, y)
            H = self.components_
            # update X_guess
            X[X < 0] = (W @ H)[X < 0]
        end_time = time.time()
        self.time = end_time - start_time
        if not self.save_space:
            self.X_guess = X
        else:
            self.X_guess = None
        return W

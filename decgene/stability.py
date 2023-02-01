""" Stability-related functions for DecGene"""

""" Drawn from Yu Group's implementation of sklearn_NMF """
""" Source: https://github.com/Yu-Group/staNMF/blob/master/staNMF/nmf_models/sklearn_nmf.py """

# define find correlation function
def findcorrelation(_A, _B):
    '''
    Construct k by k matrix of Pearson product-moment correlation
    coefficients for every combination of two columns in A and B
    Parameters
    ----------
    A : array, shape (n_features, n_components)
        first NMF solution matrix
    B : array, shape (n_features, n_components)
        second NMF solution matrix, of same dimensions as A
    Returns
    -------
    X : array shape (n_components, n_components)
        array[a][b] is the correlation between column 'a' of X
        and column 'b'
    '''

    import sklearn.preprocessing
    A_std = sklearn.preprocessing.normalize(_A, axis=0)
    B_std = sklearn.preprocessing.normalize(_B, axis=0)
    return A_std.T @ B_std

def HungarianError(_correlation):
    '''
    Compute error via Hungarian error
    based on average distance between factorization solutions
    Parameters
    ----------
    correlation: array, shape (n_components, n_components)
        cross correlation matrix
    Returns
    -------
    distM : double/float
        Hungarian distance
    '''

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n, m = _correlation.shape
    correlation = np.absolute(_correlation)  # ignore the sign of corr
    x, y = linear_sum_assignment(-_correlation)
    distM = np.mean([1 - _correlation[xx, yy] for xx, yy in zip(x, y)])
    return distM


def amariMaxError(_correlation):
    '''
    Compute what Wu et al. (2016) described as a 'amari-type error'
    based on average distance between factorization solutions
    Parameters
    ----------
    correlation: array, shape (n_components, n_components)
        cross correlation matrix
    Returns
    -------
    distM : double/float
        Amari distance
    '''

    import numpy as np

    n, m = _correlation.shape
    maxCol = np.absolute(_correlation).max(0)
    colTemp = np.mean((1-maxCol))
    maxRow = np.absolute(_correlation).max(1)
    rowTemp = np.mean((1-maxRow))
    distM = (rowTemp + colTemp)/(2)

    return distM


def instability(_folder_path, _k1, _k2, _numReplicates, _n_features=55954, _name="TBD"):
	'''
	Performs instability calculation for each K
	within the range entered
	Parameters
	----------
	tag : str
		 the name of the autoencoder model to compute the stability
	k1 : int, optional, default self.K1
		 lower bound of K to compute stability
	k2 : int, optional, default self.K2
		 upper bound of K to compute instability
	Returns
	-------
	None
	Side effects
	------------
	"instability.csv" containing instability index
	for each K between and including k1 and k2; updates
	self.instabilitydict (required for makeplot())
	'''

	import numpy as np
	import pandas as pd

	# create variables
	instabilitydict = {}
	instability_std = {}
	numPatterns = np.arange(_k1, _k2 + 1)

	# loop through each number of PPs
	for k in numPatterns:
		print("Calculating instability for " + str(k))
		# load the dictionaries
		path = _folder_path + '/K=' + str(k)
		Dhat = np.zeros((_numReplicates, _n_features, k))

		for replicate in range(_numReplicates):
			inputfilename = ('/' + _name + '_' + str(replicate) + '.npz')
			tmp = np.load(path + inputfilename, allow_pickle=True)
			PPs = tmp['PPs_tmp']
			Dhat[replicate] = PPs.T

		# compute the distance matrix between each pair of dicts
		distMat = np.zeros(shape=(_numReplicates, _numReplicates))

		for i in range(_numReplicates):
			for j in range(i, _numReplicates):
				x = Dhat[i]
				y = Dhat[j]

				CORR = findcorrelation(x, y)
				distMat[i][j] = HungarianError(CORR)
				distMat[j][i] = distMat[i][j]

		# compute the instability and the standard deviation
		instabilitydict[k] = (
				  np.sum(distMat) / (_numReplicates * (_numReplicates - 1))
		)
		# The standard deviation of the instability is tricky:
		# It is a U-statistic and in general hard to compute std.
		# Fortunately, there is a easy-to-understand upper bound.
		instability_std[k] = (np.sum(distMat ** 2)
		                      / (_numReplicates * (_numReplicates - 1))
		                      - instabilitydict[k] ** 2
		                      ) ** .5 * (2 / distMat.shape[0]) ** .5
		# write the result into csv file
		outputfile = "../output/instability_runs/instability" + _name + ".csv"
		pd.DataFrame({
			'K': [k],
			'instability': [instabilitydict[k]],
			'instability_std': [instability_std[k]],
		}).to_csv(outputfile, mode='a', header=False, index=False)
	return instabilitydict, instability_std, distMat
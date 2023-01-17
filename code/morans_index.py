""" Generate Moran's I in 3D"""


# import libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import time

###########################################
# Create adjacency matrix
###########################################

def adjacencyMatrix2D(_data):
	""" Create adjacency matrix from 2D data for Moran's I and Geary's C """

	# Determine of 2D or 3D
	if len(_data.shape) == 3:
		_3D = True
	else:
		_3D = False

	# Create variable for adjacency matrix
	w_ij = np.zeros((math.prod(_data.shape),math.prod(_data.shape)))

	# Calculate 2D adjacency matrix
	if _3D == False:

		# loop through all values of i, j
		for i in range(_data.shape[0]):
			for j in range(_data.shape[1]):
				# identify index
				w_index = i*_data.shape[1]+j

				### Record neighbors
				### Note: Assumes rook, not queen paths as neighbors

				# record left neighbor
				if j != 0:
					w_ij[w_index][w_index-1] = 1

				# record right neighbor
				if j != _data.shape[1]-1:
					w_ij[w_index][w_index+1] = 1

				# record up neighbor
				if i != 0:
					w_ij[w_index][w_index-_data.shape[1]] = 1

				# record down neighbor
				if i < _data.shape[0]-1:
					w_ij[w_index][w_index+_data.shape[1]] = 1

	return w_ij

def adjacencyMatrix3D(_data):
	""" Create adjacency matrix from 3D data for Moran's I and Geary's C """

	# Determine of 2D or 3D
	if len(_data.shape) == 3:
		_3D = True
	else:
		_3D = False

	# Create variable for adjacency matrix
	w_ij = np.zeros((math.prod(_data.shape),math.prod(_data.shape)))

	# Calculate 3D adjacency matrix
	if _3D == True:

		# loop through all values of i, j
		for i in range(_data.shape[1]):
			for j in range(_data.shape[2]):
				for k in range(_data.shape[0]):

					# identify index
					w_index = i * _data.shape[2] + k * _data.shape[1] * _data.shape[2] + j

					### Record neighbors
					### Note: Assumes rook, not queen paths as neighbors

					# record left neighbor
					if j != 0:
						w_ij[w_index][w_index-1] = 1

					# record right neighbor
					if j != _data.shape[2]-1:
						w_ij[w_index][w_index+1] = 1

					# record up neighbor
					if i != 0:
						w_ij[w_index][w_index-_data.shape[2]] = 1

					# record down neighbor
					if i < _data.shape[1]-1:
						w_ij[w_index][w_index+_data.shape[2]] = 1

					# record front neighbor
					if k != 0:
						w_ij[w_index][w_index-_data.shape[1]*_data.shape[2]] = 1

					# record back neighbor
					if k < _data.shape[0]-1:
						w_ij[w_index][w_index+_data.shape[1]*_data.shape[2]] = 1

	# return adjacency matrix
	return w_ij

def adjacencyMatrix3D_withMask(_data,_brain_mask):
	""" Create adjacency matrix from 3D data for Moran's I and Geary's C """

	# Determine of 2D or 3D
	if len(_data.shape) == 3:
		_3D = True
	else:
		_3D = False

	# # convert brain mask into a vector
	# mask = np.reshape(_brain_mask, (math.prod(_brain_mask.shape)))

	# Create variable for adjacency matrix (mask length)
	w_ij = np.zeros((math.prod(_data.shape), math.prod(_data.shape)))

	# Calculate 3D adjacency matrix
	if _3D == True:

		# loop through all values of i, j, k
		for i in range(_data.shape[1]):
			for j in range(_data.shape[2]):
				for k in range(_data.shape[0]):

					# check if index fits in mask before adding neighbors
					if _brain_mask[k,i,j] == 1:

						# identify index
						w_index = i * _data.shape[2] + k * _data.shape[1] * _data.shape[2] + j

						### Record neighbors
						### Note: Assumes rook, not queen paths as neighbors

						# record left neighbor
						if j != 0:
							w_ij[w_index][w_index-1] = 1

						# record right neighbor
						if j != _data.shape[2]-1:
							w_ij[w_index][w_index+1] = 1

						# record up neighbor
						if i != 0:
							w_ij[w_index][w_index-_data.shape[2]] = 1

						# record down neighbor
						if i < _data.shape[1]-1:
							w_ij[w_index][w_index+_data.shape[2]] = 1

						# record front neighbor
						if k != 0:
							w_ij[w_index][w_index-_data.shape[1]*_data.shape[2]] = 1

						# record back neighbor
						if k < _data.shape[0]-1:
							w_ij[w_index][w_index+_data.shape[1]*_data.shape[2]] = 1

	# return adjacency matrix with brain mask
	return w_ij

def brainMaskforAdjMat(_w, _brain_mask):
	""" Zero out any entries into w (adjacency matrix) outside of brain mask """

	# convert brain mask into a vector
	mask = np.reshape(_brain_mask, (math.prod(_brain_mask.shape)))

	print("test:", _w.shape, mask.shape)

	# apply mask to w (adjacency matrix)
	w_masked = np.matmul(_w, mask.T)

	return w_masked

###########################################
# 3. Moran's I Function
###########################################

def MoranI(_data,_brain_mask):
	""" Calculate Moran's I in 2D or 3D"""

	# Inputs to equation
	# N = math.prod(_data.shape)          # Number of pixels/voxels in data

	# N adjusted for brain mask
	N = int(np.sum(_brain_mask))

	# Create adjacency matrix for 3D data
	if len(_data.shape) == 3:
		w = adjacencyMatrix3D_withMask(_data,_brain_mask)       # Adjacency matrix for all pixels/voxels

	# Create adjacency matrix for 2D data
	if len(_data.shape) == 2:
		w = adjacencyMatrix2D(_data)       # Adjacency matrix for all pixels/voxels

	W = np.sum(w)                    # Sum of all adjacency matrix values
	x_mean = np.mean(_data)             # Mean of all pixels/voxels
	x = np.reshape(_data,math.prod(_data.shape))

	# clear variables
	_data = None

	### Calculation
	X_minus_mean = x-x_mean
	X_minus_mean = np.reshape(X_minus_mean, (len(X_minus_mean), 1))
	numerator = np.sum(np.multiply(w, np.multiply(X_minus_mean,X_minus_mean.T)))
	denominator = np.sum(np.multiply(X_minus_mean, X_minus_mean))
	I = (N / W) * (numerator / denominator)

	### Calculation, limiting to the brain map

	# # create brain mask
	# areas_atlas = np.load('mouse_coarse_structure_atlas.npy')
	# brain_mask = np.max(areas_atlas, axis=0)
	# brain_mask = np.reshape(brain_mask, math.prod(brain_mask.shape))
	#
	# # calc numerator & denominator
	# X_minus_mean = x-x_mean
	# numerator, denominator = 0, 0
	# for i in range(len(x)):
	# 	if brain_mask[i] == 1:
	# 		for j in range(len(x)):
	# 			if brain_mask[j] == 1:
	# 				numerator += w[i][j]*X_minus_mean[i]*X_minus_mean[j]
	# 				denominator += X_minus_mean[i]**2

	# # final I calculation
	# I = (np.sum(brain_mask) / W) * (numerator / denominator)

	# return Moran's Index, I
	return I

#### Function to create map of top PP for each voxel
#### To input into Moran's I calc
#### Note: Will this help? TBD by how much

def PPmap(_PPs):
	""" Map PPs for Moran's I calculation"""
	PP_map = np.zeros((_PPs.shape[1:]))

	# loop through each voxel
	for i in range(PP_map.shape[0]):
		for j in range(PP_map.shape[1]):
			for k in range(PP_map.shape[2]):
				max_PP = np.argmax(_PPs[:, i, j, k])
				if _PPs[max_PP,i,j,k] > 0:
					PP_map[i,j,k] = max_PP
				else:
					PP_map[i, j, k] = -1

	# clear variables
	_PPs, max_PP = None, None

	return PP_map

#### Function to remove surrounding empty layers from PP
def removeEmptyLayers(_PP):
	while np.sum(_PP[0,:,:]) == 0:
		_PP = np.delete(_PP, 0, 0)
	while np.sum(_PP[-1,:,:]) == 0:
		_PP = np.delete(_PP, -1, 0)
	while np.sum(_PP[:,0,:]) == 0:
		_PP = np.delete(_PP, 0, 1)
	while np.sum(_PP[:,-1,:]) == 0:
		_PP = np.delete(_PP, -1, 1)
	while np.sum(_PP[:,:,0]) == 0:
		_PP = np.delete(_PP, 0, 2)
	while np.sum(_PP[:,:,-1]) == 0:
		_PP = np.delete(_PP, -1, 2)
	# print("New PP Shape:", _PP.shape, np.sum(_PP))
	return _PP

#### Calc Moran's I for PP sets
def MoranIforPPs(_PPs, _removeEmptyLayers, _brain_mask):

	# Create PP map
	num_PPs = len(_PPs)
	PP_map = PPmap(_PPs)

	I_list = []
	for p in range(num_PPs):

		# Process map into 1 (for PP p) or 0 (for all other regions)
		PP_map_p = np.where(PP_map == p, 1, 0)

		# # Remove empty layers around PP
		# if _removeEmptyLayers:
		# 	PP_map_p = removeEmptyLayers(PP_map_p)

		# Run Moran's I
		I = MoranI(PP_map_p,_brain_mask)
		if np.isnan(I): # if NaN, append as 1 as there is only 1 class present
			I_list.append(1)
		else:
			I_list.append(I)

		# Clear variables
		PP_map_p = None

	# Output list of Moran's I values
	return I_list

def chunkMoranI(_PPs, _brain_mask):
	""" Break up Moran's I into smaller chunks for calc & average them up"""
	chunk_size = 10 # in voxels, e.g. 10x10x10 chunks
	chunks = int(_PPs.shape[1]/chunk_size)+1
	I_cumulative = np.zeros((_PPs.shape[0]))
	for i in range(chunks):
		I_tmp = np.asarray(MoranIforPPs(_PPs[:,i*chunk_size:(i+1)*chunk_size,:,:],
		                                False,
		                                brain_mask[i*chunk_size:(i+1)*chunk_size,:,:]))
		I_cumulative = I_cumulative * (i+1) / (i+2)
		I_tmp = I_tmp / (i+2)
		I_cumulative += I_tmp

	return I_cumulative


### Run Moran's I for n different boostrapped PPs
def runManyMoranI(_n, _DecGeneOrPCA, _PPs, _mask, _downsample):
	""" Gets Moran's I for n runs of bootstrapped PPs"""

	# create variable to store all runs
	I_manyRuns = np.zeros(( _n, _PPs.shape[0]))

	# create support for getting PPs into 3D
	areas_atlas = np.load('../data/mouse_coarse_structure_atlas.npy')
	support = np.sum(areas_atlas, 0) > 0

	# Downsample mask
	for d in range(3):
		del_list = []
		for i in range(_mask.shape[d]):
			if (i) % _downsample == 0:
				del_list.append(i)
		_mask = np.delete(_mask, del_list, axis=d)

	# set variable for DecGene or PCA
	if _DecGeneOrPCA == "DecGene":
		DGorPCA = "DG_PP_"
		path_tmp = "../output/sims_DG_PCAvsCCF/DG_PPs"
	elif _DecGeneOrPCA == "PCA":
		DGorPCA = "PCA_PP_"
		path_tmp = "../output/sims_DG_PCAvsCCF/PCA_PPs"
	else:
		print("Possible error in PCA or DecGene?")

	for n in range(_n):

		# for debugging to track time
		start = time.time()

		# load PPs
		file_tmp = DGorPCA + str(n) + '.npz'
		os.path.join(path_tmp, file_tmp)
		tmp = np.load(os.path.join(path_tmp, file_tmp))
		PPs_tmp = tmp["PPs_tmp"]

		# get PPs into 3D
		PPs_3d_tmp = np.zeros((11, 66, 40, 57))

		if _DecGeneOrPCA == "DecGene":
			PPs_3d_tmp[:, support] = PPs_tmp * (PPs_tmp > 0.05)  # note: this is >0.05 for DecGene, >0 for PCA
		else:
			PPs_3d_tmp[:, support] = PPs_tmp * (PPs_tmp > 0)  # note: this is >0.05 for DecGene, >0 for PCA

		# Downsample PPs by deleting extra values
		for d in range(3):
			del_list = []
			for i in range(PPs_3d_tmp.shape[d+1]):
				if (i) % _downsample == 0:
					del_list.append(i)
			PPs_3d_tmp = np.delete(PPs_3d_tmp, del_list, axis=d+1)

		# run Moran's I
		I_PPs = MoranIforPPs(PPs_3d_tmp, False, _mask) # if can do full calculation without chunking up
		# I_PPs = chunkMoranI(PPs_3d_tmp, _mask) # if need to chunk up the calculation

		# save to array with all runs
		I_manyRuns[n] = I_PPs

		# debugging
		end = time.time()
		print("---Finished run", DGorPCA, n+1, "out of", _n, "runs.")
		print("Time to run:", end-start)

	return I_manyRuns

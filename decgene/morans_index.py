""" Calculate Moran's I in 3D"""

# import libraries
import math
import numpy as np


# Create adjacency matrix in 3D

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

	# Create adjacency matrix
	w = adjacencyMatrix3D_withMask(_data,_brain_mask)       # Adjacency matrix for all pixels/voxels

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

	return I


# Function to create map of top PP for each voxel

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


#### Calc Moran's I for PP sets

def MoranIforPPs(_PPs, _brain_mask):

	# Create PP map
	num_PPs = len(_PPs)
	PP_map = PPmap(_PPs)

	I_list = []
	for p in range(num_PPs):

		# Process map into 1 (for PP p) or 0 (for all other regions)
		PP_map_p = np.where(PP_map == p, 1, 0)

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

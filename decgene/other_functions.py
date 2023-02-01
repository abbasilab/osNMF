""" Other functions for DecGene"""


def correlation_map_with_CCF(PPs, original_shape, plot=True,
                             order_type=1, area_order=None,
                             put_last_k=0, save_fig=False, save_index=None):
	''' Compare PPs with the standard ABA CCF.
	'''
	# transform PPs to 4d tensor
	PPs_3d = np.zeros([PPs.shape[0]] + original_shape[1:].tolist())
	num_pps = PPs.shape[0]
	for i in range(PPs.shape[0]):
		p2 = np.reshape(PPs[i, :], original_shape[1:])
		PPs_3d[i, :, :, :] = p2
	# load ABA CCF coarse
	areas_atlas = np.load('../data/mouse_coarse_structure_atlas.npy')
	mouse_coarse_df = pd.read_pickle('../data/mouse_coarse_df')
	if area_order != None:
		cor_mat = np.corrcoef(
			np.vstack([areas_atlas.reshape(12, -1)[np.array(area_order)], PPs_3d.reshape(num_pps, -1)]))[
		          :areas_atlas.shape[0], areas_atlas.shape[0]:]
	else:
		cor_mat = np.corrcoef(np.vstack([areas_atlas.reshape(12, -1), PPs_3d.reshape(num_pps, -1)]))[
		          :areas_atlas.shape[0], areas_atlas.shape[0]:]

	if order_type == 1:
		rows, cols = linear_sum_assignment(-np.abs(cor_mat))
		factor_order = list(cols) + [x for x in range(num_pps) if x not in cols]
	elif order_type == 2:
		cols = np.argmax(np.abs(cor_mat), 0)
		if put_last_k > 0:
			# put the poorly fitted patterns at the last.
			best_fits = [abs(cor_mat[y, x]) for x, y in enumerate(cols)]
			orders = np.argsort(best_fits)
			for i in range(put_last_k):
				cols[orders[i]] = max(cols)
		factor_order = np.argsort(
			[10 * x - abs(cor_mat[x, i]) for i, x in enumerate(cols.tolist())])  # first sort by x, then sort by the value

	if plot:
		fig = plt.gcf()
		plt.figure(figsize=(10, 9))
		plt.imshow(np.abs(cor_mat[:, factor_order]).tolist(), cmap='YlOrRd')
		if area_order is None:
			plt.yticks(np.arange(12), (mouse_coarse_df.iloc[:]['name'].tolist()))
		else:
			plt.yticks(np.arange(12), (mouse_coarse_df.iloc[area_order]['name'].tolist()))
		plt.ylim([-0.5, 11.5])
		plt.gca().invert_yaxis()
		plt.xticks(range(num_pps), factor_order)
		plt.title('Correlation Coefficient')
		plt.xlabel('Principle Patterns')
		plt.colorbar(shrink=.6)
		plt.savefig('../figures/corr_coeff.png')
		plotname = "corr_map.png"
		plt.savefig(plotname, dpi=300)
		plt.show()
	return np.abs(cor_mat[:, factor_order])


def geneReconstructionAcccuracy(_PPs, _coeffs, _filtered_data):
	""" function create gene-by-gene reconstruction accuracy """

	import numpy as np

	gene_rec_accuracy_lst = []

	# loop through each gene
	for i in range(len(_coeffs)):

		# reconstruct gene
		gene_rec = np.matmul(_coeffs[i], _PPs)

		# get actual gene value from ABA data
		gene_actual = np.maximum(_filtered_data[i], 0)

		# estimate Pearson corr coeff between gene reconstruction and original data
		corr = np.corrcoef(gene_rec, gene_actual)[0][1]

		# set to 0 if NaN
		if np.isnan(corr):
			corr = 0

		# add calculated correlation to list
		gene_rec_accuracy_lst.append(corr)

	return gene_rec_accuracy_lst


def show_graph_with_labels(adjacency_matrix, mylabels, colors, title=""):
	""" Function to generate graphs of correlation networks """
	# plt.figure(figsize=(30,30))
	fig1, f1_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(25, 14))
	rows, cols = np.where(adjacency_matrix > 0)
	edges = [(x, y) for x, y in zip(rows.tolist(), cols.tolist()) if x > y]
	edge_colors = np.array([adjacency_matrix[edge[0], edge[1]] for edge in edges])
	gr = nx.Graph()
	gr.add_edges_from(edges)

	remove = [node for node, degree in gr.degree() if degree <= 0.5]
	colors = [x for x, y in zip(colors, gr.degree()) if y[1] > 0.5]
	edge_colors = [y for x, y in zip(edges, edge_colors) if gr.degree()[x[0]] > 0.5 and gr.degree()[x[1]] > 0.5]

	gr.remove_nodes_from(remove)
	nodes = set(ind for ind, name in gr.degree())
	pos = nx.circular_layout(gr)
	f1_axes[1, 1].set_xlim([-2, 2])
	plt.ylim([-2, 2])
	nx.draw(
		gr,
		pos,
		node_size=300,
		labels={k: v for k, v in mylabels.items() if k in gr},
		node_color=colors,
		with_labels=False,
		cmap=plt.cm.copper_r,
	)
	nx.draw_networkx_edges(
		gr,
		pos,
		gr.edges(),
		edge_color=edge_colors,
		edge_cmap=plt.cm.Greys,
	)
	description = nx.draw_networkx_labels(gr, {k: (v[0] * 1.25, v[1] * 1.25) for k, v in pos.items()},
	                                      labels={k: v for k, v in mylabels.items() if k in gr})
	sm = plt.cm.ScalarMappable(cmap=plt.cm.copper_r, norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors)))
	sm._A = []
	plt.colorbar(sm)
	plt.title(title)
	plt.savefig("../figures/network_names{}.png".format(title), dpi=300)
	plt.show()


def visualize_gene(geneA, template=0, colorA='green', missing_mask=True):
	""" define_function a function that shows two genes together """

	import numpy as np
	from PIL import Image, ImageFilter

	geneA = np.maximum(geneA, 0)

	color_map = {'red': 0, 'green': 1, 'blue': 2}
	colorA = color_map[colorA]
	if missing_mask:
		geneA = geneA > 0

	_, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5))

	x_dim, y_dim, z_dim = geneA.shape

	if np.sum(template) == 0:
		template = np.zeros((x_dim, y_dim, z_dim))

	out1 = np.zeros((y_dim, z_dim, 3))
	out1[:, :, colorA] = np.sum(geneA, 0)
	out1[:, :, colorA] /= np.max(out1[:, :, colorA])

	# to make yellow for PCA
	out1[:, :, 1] = np.sum(geneA, 0)
	out1[:, :, 1] /= np.max(out1[:, :, 1])

	image = Image.fromarray(np.uint8(cm.gist_earth(np.sum(template, 0)) * 255))
	image = image.filter(ImageFilter.MaxFilter(3))
	image = image.filter(ImageFilter.MinFilter(3))
	image = image.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
	                                                 -1, -1, -1, -1), 1, 0))
	support_x = np.array(image.filter(ImageFilter.FIND_EDGES))[:, :, 0]

	for rgb_ch in range(3):
		out1[:, :, rgb_ch] = out1[:, :, rgb_ch] + support_x
	#         out1[:,:,1] = support_x
	c.imshow(out1)
	# plt.colorbar()

	out2 = np.zeros((x_dim, z_dim, 3))
	out2[:, :, colorA] = np.sum(geneA, 1)
	out2[:, :, colorA] /= np.max(out2[:, :, colorA])

	# to make yellow for PCA
	out2[:, :, 1] = np.sum(geneA, 1)
	out2[:, :, 1] /= np.max(out2[:, :, 1])

	image = Image.fromarray(np.uint8(cm.gist_earth(np.sum(template, 1)) * 255))
	image = image.filter(ImageFilter.MaxFilter(5))
	image = image.filter(ImageFilter.MinFilter(5))
	image = image.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
	                                                 -1, -1, -1, -1), 1, 0))
	support_y = np.array(image.filter(ImageFilter.FIND_EDGES))[:, :, 0]

	for rgb_ch in range(3):
		out2[:, :, rgb_ch] = out2[:, :, rgb_ch] + support_y
	#         out2[:,:,1] = support_y
	b.imshow(out2)
	# plt.colorbar()

	plt.figure()

	out3 = np.zeros((y_dim, x_dim, 3))
	out3[:, :, colorA] = np.sum(geneA, 2).T
	out3[:, :, colorA] /= np.max(out3[:, :, colorA])

	# to make yellow
	out3[:, :, 1] = np.sum(geneA, 2).T
	out3[:, :, 1] /= np.max(out3[:, :, 1])

	image = Image.fromarray(np.uint8(cm.gist_earth(np.sum(template, 2)) * 255))
	image = image.filter(ImageFilter.MaxFilter(3))
	image = image.filter(ImageFilter.MinFilter(3))
	image = image.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
	                                                 -1, -1, -1, -1), 1, 0))
	support_z = np.array(image.filter(ImageFilter.FIND_EDGES))[:, :, 0]

	for rgb_ch in range(3):
		out3[:, :, rgb_ch] = out3[:, :, rgb_ch] + support_z.T
	#         out3[:,:,1] = support_z.T

	a.imshow(out3)

	#     a.text(1, 6, 'PP{}'.format(i), color='w', fontsize=30)
	#     b.text(1, 6, 'PP{}'.format(i), color='w', fontsize=30)
	#     c.text(1, 6, 'PP{}'.format(i), color='w', fontsize=30)

	a.axes.get_xaxis().set_visible(False)
	a.axes.get_yaxis().set_visible(False)
	b.axes.get_xaxis().set_visible(False)
	b.axes.get_yaxis().set_visible(False)
	c.axes.get_xaxis().set_visible(False)
	c.axes.get_yaxis().set_visible(False)

	plt.show()


def filter_genes(coefs, pps, threshold = .99):
    large_coefs = np.max(coefs[:, pps], 1) / np.sum(coefs, 1)
    return large_coefs > np.quantile(large_coefs, threshold)


def weighted_correlation(A, weights, demean=True):
    '''
    Compute the weighted correlation between columns of A using the weights.
    '''
    if demean:
        A = A - np.mean(A, 0, keepdims=True)
    mean_A = 0#np.sum(A * weights, 1, keepdims=True) / np.sum(weights)
    cov = (A - mean_A) @ (A - mean_A).T
    corr = np.diag(np.diag(cov) ** -.5) @ cov @ np.diag(np.diag(cov)**-.5)
    return corr
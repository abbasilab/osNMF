""" Other functions for DecGene"""

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

	import matplotlib.pyplot as plt
	import numpy as np

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
	plt.show()


def visualize_gene(geneA, template=0, colorA='green', missing_mask=True):
	""" define a function that shows two genes together """

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
	c.imshow(out1)

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
	b.imshow(out2)

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

	a.imshow(out3)

	a.axes.get_xaxis().set_visible(False)
	a.axes.get_yaxis().set_visible(False)
	b.axes.get_xaxis().set_visible(False)
	b.axes.get_yaxis().set_visible(False)
	c.axes.get_xaxis().set_visible(False)
	c.axes.get_yaxis().set_visible(False)

	plt.show()


def filter_genes(coefs, pps, threshold = .99):
	import numpy as np
	large_coefs = np.max(coefs[:, pps], 1) / np.sum(coefs, 1)
	return large_coefs > np.quantile(large_coefs, threshold)


def weighted_correlation(A, weights, demean=True):
    '''
    Compute the weighted correlation between columns of A using the weights.
    '''

    import numpy as np

    if demean:
        A = A - np.mean(A, 0, keepdims=True)
    mean_A = 0
    cov = (A - mean_A) @ (A - mean_A).T
    corr = np.diag(np.diag(cov) ** -.5) @ cov @ np.diag(np.diag(cov)**-.5)
    return corr
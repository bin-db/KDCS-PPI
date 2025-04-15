import copy
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import torch
import networkx as nx
import numpy as np
from sknetwork.utils import membership_matrix
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def process_cf(data, index_arch, index_target):
    mat_f = get_mat_f(data, False)
    # simi_mat is a num_nodes * num_nodes matrix, where each element represents the similarity between the corresponding nodes.
    simi_mat = get_simi_mat(data.x, 'euclidean')

    return simi_mat


def get_mat_f(data, self_loop=False):
    num_edges = data.num_edges
    num_nodes = data.num_nodes
    sparse_mat = coo_matrix((torch.ones(num_edges), (data.edge_index[0], data.edge_index[1])),
                            shape=(num_nodes, num_nodes))

    if not self_loop:
        sparse_mat.setdiag(0)
        sparse_mat.eliminate_zeros()

    return kcore(sparse_mat)


def kcore(adj):
    g = nx.from_scipy_sparse_matrix(adj)
    g.remove_edges_from(nx.selfloop_edges(g))

    labels = np.array(list(nx.algorithms.core.core_number(g).values())) - 1
    mem_mat = membership_matrix(labels)
    t = (mem_mat @ mem_mat.T).astype(int)

    return t


def get_simi_mat(feats, dist='euclidean'):
    if dist == 'cosine':
        simi_mat = cosine_similarity(feats, feats)
    elif dist == 'euclidean':
        simi_mat = cdist(feats, feats, 'euclidean')

    return simi_mat

import copy
import os
import scipy.sparse as sp
import networkx as nx
import numpy as np
import torch
from sknetwork.clustering import KMeans
from sknetwork.utils import membership_matrix
from sknetwork.embedding import Spectral
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
import random
import pickle
from torch.nn import functional as F
from sknetwork.hierarchy import Ward, cut_straight


def load_cf_data(T_file, adj, feat):
    """
        T_f: Adjacency matrix indicating whether two nodes belong to the same community
        T_cf: has 'adj_cf' undergone counterfactual processing
        adj_cf: do nodes i, j have an edge in the counterfactual scenario
        sampling_prob_mat: the similarity between nodes i, j with their counterfactual nodes: the smaller the value, the
                            higher the similarity.
    """
    if os.path.exists(T_file):
        T_f, T_cf, adj_cf, sampling_prob_mat = pickle.load(open(T_file, 'rb'))
    else:
        T_f, T_cf, adj_cf, sampling_prob_mat = process_CF(adj, feat)
        T_cf = sp.csr_matrix(T_cf).astype(np.int64)
        adj_cf = sp.csr_matrix(adj_cf).astype(np.int64)
        sampling_prob_mat = sp.csr_matrix(sampling_prob_mat)

        pickle.dump((T_f, T_cf, adj_cf, sampling_prob_mat), open(T_file, 'wb'))

    return T_f, T_cf, adj_cf, sampling_prob_mat


def is_symmetric(mat):
    arr_mat = mat.toarray()
    return np.allclose(arr_mat, arr_mat.T)


def load_target_nodes_data(nodes_file, nodes_num, adj, adj_cf, T_f, T_cf, sampling_prob_mat, device, sampling_nodes_num=3):
    if os.path.exists(nodes_file):
        pos_nodes_tensor, neg_nodes_tensor, pos_nodes_prob_tensor, neg_nodes_prob_tensor = pickle.load(open(nodes_file, 'rb'))
    else:
        pos_nodes_tensor = torch.empty(0).to(device)
        neg_nodes_tensor = torch.empty(0).to(device)
        pos_nodes_prob_tensor = torch.empty(0).to(device)
        neg_nodes_prob_tensor = torch.empty(0).to(device)
        adj = torch.tensor(adj).to(device)

        # Convert T_f, T_cf, adj_cf, and sampling_prob_mat from upper triangular matrices to symmetric matrices
        T_f = torch.Tensor(T_f.astype(np.int64).toarray()).to(device)
        T_cf = torch.Tensor(T_cf.astype(np.int64).toarray()).to(device)
        T_cf = T_cf + T_cf.transpose(0, 1)
        adj_cf = torch.Tensor(adj_cf.astype(np.int64).toarray()).to(device)
        adj_cf = adj_cf + adj_cf.transpose(0, 1)
        sampling_prob_mat = torch.Tensor(sampling_prob_mat.toarray()).to(device)
        sampling_prob_mat = sampling_prob_mat + sampling_prob_mat.transpose(0, 1)

        cf_sampling_pos_nums = list(-1 for _ in range(nodes_num))
        cf_sampling_neg_nums = list(-1 for _ in range(nodes_num))
        for arch_node in tqdm(range(nodes_num), desc='processing target_nodes'):
            pos_t_nodes = []
            neg_t_nodes = []
            pos_t_nodes_prob = []
            neg_t_nodes_prob = []
            for i in range(nodes_num):
                if i != arch_node and T_f[arch_node, i] != T_cf[arch_node, i]:
                    if len(pos_t_nodes) < sampling_nodes_num and (adj[arch_node, i] == 1 and adj_cf[arch_node, i] == 1):
                        pos_t_nodes.append(i)
                        pos_t_nodes_prob.append(sampling_prob_mat[arch_node, i])
                    elif len(neg_t_nodes) < sampling_nodes_num and (adj[arch_node, i] == 0 and adj_cf[arch_node, i] == 0):
                        neg_t_nodes.append(i)
                        neg_t_nodes_prob.append(sampling_prob_mat[arch_node, i])

                if len(pos_t_nodes) == sampling_nodes_num and len(neg_t_nodes) == sampling_nodes_num:
                    break

            cf_sampling_pos_nums[arch_node] = len(pos_t_nodes)
            cf_sampling_neg_nums[arch_node] = len(neg_t_nodes)

            # If the number of positive and negative samples does not meet sampling_nodes_num, fill with random values
            if len(pos_t_nodes) < sampling_nodes_num:
                pos_t_nodes += add_samples(arch_node, sampling_nodes_num, nodes_num, 1, pos_t_nodes, adj)
                # pos_t_nodes += [random.choice(list(filter(lambda x: x not in pos_t_nodes and x != arch_node,
                #                                           range(nodes_num)))) for _ in
                #                range(sampling_nodes_num - len(pos_t_nodes))]
                if pos_t_nodes_prob:
                    pos_t_nodes_prob += [max(pos_t_nodes_prob) * 2] * (sampling_nodes_num - len(pos_t_nodes_prob))
                else:
                    pos_t_nodes_prob = [0.0 for _ in range(sampling_nodes_num)]

            if len(neg_t_nodes) < sampling_nodes_num:
                neg_t_nodes += add_samples(arch_node, sampling_nodes_num, nodes_num, 0, neg_t_nodes, adj)

                if neg_t_nodes_prob:
                    neg_t_nodes_prob += [max(neg_t_nodes_prob) * 2] * (sampling_nodes_num - len(neg_t_nodes_prob))
                else:
                    neg_t_nodes_prob = [0.0 for _ in range(sampling_nodes_num)]

            pos_t_nodes = torch.tensor(pos_t_nodes).to(device)
            neg_t_nodes = torch.tensor(neg_t_nodes).to(device)
            pos_t_nodes_prob = torch.tensor(pos_t_nodes_prob).to(device)
            neg_t_nodes_prob = torch.tensor(neg_t_nodes_prob).to(device)

            if pos_nodes_tensor.numel() == 0:
                pos_nodes_tensor = pos_t_nodes.unsqueeze(0)
                pos_nodes_prob_tensor = pos_t_nodes_prob.unsqueeze(0)
            else:
                pos_nodes_tensor = torch.cat((pos_nodes_tensor, pos_t_nodes.unsqueeze(0)), dim=0).to(device)
                pos_nodes_prob_tensor = torch.cat((pos_nodes_prob_tensor, pos_t_nodes_prob.unsqueeze(0)), dim=0).to(
                    device)
            if neg_nodes_tensor.numel() == 0:
                neg_nodes_tensor = neg_t_nodes.unsqueeze(0)
                neg_nodes_prob_tensor = neg_t_nodes_prob.unsqueeze(0)
            else:
                neg_nodes_tensor = torch.cat((neg_nodes_tensor, neg_t_nodes.unsqueeze(0)), dim=0).to(device)
                neg_nodes_prob_tensor = torch.cat((neg_nodes_prob_tensor, neg_t_nodes_prob.unsqueeze(0)), dim=0).to(
                    device)

        pos_nodes_prob_tensor, neg_nodes_prob_tensor = process_prob(pos_nodes_prob_tensor, neg_nodes_prob_tensor)

        pickle.dump((pos_nodes_tensor, neg_nodes_tensor, pos_nodes_prob_tensor, neg_nodes_prob_tensor),
                    open(nodes_file, 'wb'))

    return pos_nodes_tensor, neg_nodes_tensor, pos_nodes_prob_tensor, neg_nodes_prob_tensor


def counter_nodes_cf(cf_nodes):
    cf_nodes_dict = {}
    for num in cf_nodes:
        if num in cf_nodes_dict:
            cf_nodes_dict[num] += 1
        else:
            cf_nodes_dict[num] = 1

    return cf_nodes_dict


def add_samples(arch_node, sampling_nodes_num, nodes_num, is_pos, t_nodes, adj):
    """
    Parameters
    ----------
    # arch_nodes: arch_node
    # sample_nodes_num: the num of needed pos/neg samples
    # nodes_num: the num of nodes in graph
    # is_pos: 1-find pos target nodes; 0-find neg target nodes
    # t_nodes: the list of pos/neg samples finding by counterfactual samples selecting

    Returns
    -------
    """
    total_nodes = list(torch.where(adj[arch_node] == is_pos)[0])
    total_nodes = [x.item() for x in total_nodes]
    if len(total_nodes) == 0:
        return random.sample(range(nodes_num), sampling_nodes_num - len(t_nodes))

    if len(total_nodes) - 1 >= sampling_nodes_num:
        select_nodes = [random.choice(list(filter(lambda x: x not in t_nodes and x != arch_node, total_nodes)))
                        for _ in range(sampling_nodes_num - len(t_nodes))]
    else:
        # If total_nodes - 1 (excluding arch_node) is less than sampling_nodes_num
        # First, fill select_nodes with all total_nodes, then randomly select duplicates to expand select_nodes to sampling_nodes_num
        select_nodes = [x for x in total_nodes if x != arch_node and x not in t_nodes]
        index = torch.randint(0, len(total_nodes), (sampling_nodes_num - len(t_nodes) - len(select_nodes),))
        select_nodes += [total_nodes[i] for i in index]

    return select_nodes


def process_prob(p_nodes_prob, n_nodes_prob):
    p_nodes_prob = torch.neg(p_nodes_prob)
    n_nodes_prob = torch.neg(n_nodes_prob)

    p_nodes_prob = F.softmax(p_nodes_prob, dim=1)
    n_nodes_prob = F.softmax(n_nodes_prob, dim=1)

    return p_nodes_prob, n_nodes_prob


def process_CF(adj, features):
    T_f = get_t(adj, 'kcore', False)
    T_cf, adj_cf, sampling_prob_mat = get_CF(adj, features, T_f, 'euclidean', thresh=50, n_workers=2)
    return T_f, T_cf, adj_cf, sampling_prob_mat


def get_t(adj_mat, method='hierarchy', self_loop=False):
    adj = copy.deepcopy(adj_mat)
    sparse_adj = sp.csr_matrix(adj)
    if not self_loop:
        sparse_adj.setdiag(0)
        sparse_adj.eliminate_zeros()

    if method == 'kcore':
        T = kcore(sparse_adj)
    elif method == 'hierarchy':
        T = ward_hierarchy(adj, 2)

    return T


def get_CF(adj, node_embs, T_f, dist='euclidean', thresh=50, n_workers=50):
    node_embs = normalize(node_embs, norm='l1', axis=1)
    if dist == 'cosine':
        simi_mat = cosine_similarity(node_embs, node_embs)
    elif dist == 'euclidean':
        # Euclidean distance
        simi_mat = cdist(node_embs, node_embs, 'euclidean')

    # Return the median of all elements in simi_mat
    thresh = np.percentile(simi_mat, thresh)

    # Fill diagonal elements of simi_mat with (max value of simi_mat + 1)
    np.fill_diagonal(simi_mat, np.max(simi_mat) + 1)

    node_nns = np.argsort(simi_mat, axis=1)

    # node_pairs contains all combinations of node pairs
    # e.g., for nodes 1, 2, 3, node_pairs = [(1, 2), (1, 3), (2, 3)]
    node_pairs = list(combinations(range(adj.shape[0]), 2))
    print('This step may be slow, please adjust args.n_workers according to your machine')
    pool = Pool(n_workers)
    batches = np.array_split(node_pairs, n_workers)

    results = pool.map(get_CF_single, [(adj, simi_mat, node_nns, T_f, thresh, np_batch, True) for np_batch in batches])
    results = list(zip(*results))

    T_cf = np.add.reduce(results[0])
    adj_cf = np.add.reduce(results[1])
    sampling_prob_mat = np.add.reduce(results[2])

    return T_cf, adj_cf, sampling_prob_mat


def get_CF_single(params):
    """
    single process for getting CF edges
    """
    adj, simi_mat, node_nns, T_f, thresh, node_pairs, verbose = params

    T_cf = np.zeros(adj.shape)
    adj_cf = np.zeros(adj.shape)
    sampling_prob_mat = np.zeros(adj.shape)

    for a, b in tqdm(node_pairs, desc='processing node pairs'):
        nns_a = node_nns[a]
        nns_b = node_nns[b]

        i, j = 0, 0

        while i < len(nns_a) - 1 and j < len(nns_b) - 1:
            # If there is no pair with distance less than 2 * thresh between (a, b),  then T_cf and adj_cf remain the same as the original graph
            # Larger distance means lower similarity
            if simi_mat[a, nns_a[i]] + simi_mat[b, nns_b[j]] > 2 * thresh:
                T_cf[a, b] = T_f[a, b]
                adj_cf[a, b] = adj[a, b]
                break

            # There exists a pair (i, j) whose distance is less than 2 * thresh from (a, b),
            # and i and j belong to different communities than a and b
            if T_f[nns_a[i], nns_b[j]] != T_f[a, b]:
                T_cf[a, b] = 1 - T_f[a, b]
                adj_cf[a, b] = adj[nns_a[i], nns_b[j]]
                sampling_prob_mat[a, b] = simi_mat[a, nns_a[i]] + simi_mat[b, nns_b[j]]
                break

            if simi_mat[a, nns_a[i+1]] < simi_mat[b, nns_b[j+1]]:
                i += 1
            else:
                j += 1

    return T_cf, adj_cf, sampling_prob_mat


def kcore(adj):
    G = nx.from_scipy_sparse_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    labels = np.array(list(nx.algorithms.core.core_number(G).values())) - 1
    # mem_mat is a matrix with size n * c, where n is number of nodes, and c is community number
    # in mem_mat，the i-th row represents whether the i-th node belong to the j-th community, with mem_mat[i, j]=1
    # if it does belong, and 0 otherwise
    mem_mat = membership_matrix(labels)
    # T is an n*n matrix where T[i, j] indicates whether the i-th and j-th nodes belong to the same community
    T = (mem_mat @ mem_mat.T).astype(int)
    return T


def spectral_clustering(adj, k):
    kmeans = KMeans(n_clusters=k, embedding_method=Spectral(256))
    labels = kmeans.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T


def ward_hierarchy(adj, k):
    print('use ward_hierarchy')
    ward = Ward()
    dendrogram = ward.fit_transform(adj)
    labels = cut_straight(dendrogram, k)
    print(f'labels is {labels}')
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T


def select_target_nodes(pos_t_nodes, neg_t_nodes, pos_t_nodes_prob, neg_t_nodes_prob, node_perm, select_nodes_num=2):
    device = pos_t_nodes.device
    pos_nodes_sub = pos_t_nodes[node_perm]
    neg_nodes_sub = neg_t_nodes[node_perm]
    pos_nodes_prob_sub = pos_t_nodes_prob[node_perm]
    neg_nodes_prob_sub = neg_t_nodes_prob[node_perm]

    # Select nodes based on probability
    pos_select_indices = torch.multinomial(pos_nodes_prob_sub, select_nodes_num, replacement=False).to(device)
    neg_select_indices = torch.multinomial(neg_nodes_prob_sub, select_nodes_num, replacement=False).to(device)

    pos_samples = pos_nodes_sub[torch.arange(pos_nodes_sub.size(0)).unsqueeze(1), pos_select_indices].to(device)
    node_perm_tensor = torch.tensor(node_perm).unsqueeze(1).to(device)
    pos_samples = torch.cat((node_perm_tensor, pos_samples), dim=1)

    neg_samples = neg_nodes_sub[torch.arange(neg_nodes_sub.size(0)).unsqueeze(1), neg_select_indices].to(device)

    return pos_samples, neg_samples

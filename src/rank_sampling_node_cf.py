import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def classify_sample_nodes(data, index_mat, pos_batch, neg_batch, device):
    pos_simi_edge = get_cf_edge_info(data, pos_batch, index_mat, device)
    neg_simi_edge = get_cf_edge_info(data, torch.cat((pos_batch[:, 0].unsqueeze(1), neg_batch), dim=1), index_mat, device)
    return classify_edge_type(data, pos_batch, neg_batch, pos_simi_edge, neg_simi_edge)


def get_cf_edge_info(data, nodes_batch, index_mat, device):
    """
        nodes_batch: shape is batch_size * nodes
            nodes_batch[:, 0]: represents the anchor node
            nodes_batch[:, 1:]: represents the nodes traversed during the random walk starting from the anchor node, possibly traversing multiple times
        index_mat: The i-th row represents the similarity between node i and other nodes, [i, 1] represents the index of the most similar node, [i, 2] represents the index of the second most similar node, and so on.

        edge_info: represents whether there is an edge between the node in nodes_batch[i, 1:] and the anchor node nodes_batch[i, 0] under counterfactual conditions
"""
    new_nodes = torch.empty(0, dtype=torch.long).to(device)
    for i in range(nodes_batch.size(0)):
        row_node_index = index_mat[nodes_batch[i], 1].to(device)
        new_nodes = torch.cat((new_nodes, row_node_index.unsqueeze(0)), dim=0).to(device)

    arch_nodes_simi = new_nodes[:, 0].to(device)
    target_nodes_simi = new_nodes[:, 1:].to(device)
    cf_edge_info = is_exist_edge(data, arch_nodes_simi, target_nodes_simi, device)

    return cf_edge_info


def is_exist_edge(data, arch_nodes, target_nodes, device):
    edge_info = torch.empty(0, dtype=torch.int).to(device)
    for i in range(len(arch_nodes)):
        t_nodes = target_nodes[i].to(device)
        edges = []
        for t_node in t_nodes:
            edges.append(int(contains_edge(data, arch_nodes[i], t_node)))
        edge_info = torch.cat((edge_info, torch.tensor(edges, dtype=torch.int).unsqueeze(0).to(device)), 0)

    return edge_info


def contains_edge(data, arch_node, t_node):
    is_contains_edge = contains_arc_edge(data.edge_index[0], data.edge_index[1], arch_node,
                                         t_node) or contains_arc_edge(data.edge_index[1], data.edge_index[0], arch_node,
                                                                      t_node)

    return is_contains_edge


def contains_arc_edge(edge_list_1, edge_list_2, arch_node, t_node):
    indices = torch.where(edge_list_1 == arch_node)
    t_edge_nodes = edge_list_2[indices]

    return t_node in t_edge_nodes


def get_simi_mat(x):
    """
        x: feature matrix for all samples
        simi_mat: similarity matrix
        sorted_indices[i, :]: represents the sorted nodes based on similarity with the i-th sample
                              where sorted_indices[i, 1] is the index of the sample most similar to the i-th sample, excluding itself
    """
    x_np = x.detach().cpu().numpy()
    simi_mat = cosine_similarity(x_np, x_np)
    sorted_indices = np.argsort(-simi_mat, axis=1)
    simi_mat = torch.tensor(simi_mat).to(x.device)
    sorted_indices = torch.tensor(sorted_indices).to(x.device)

    return simi_mat, sorted_indices


def label_edge_type(edge_type, indices, type_num):
    for idx in indices:
        edge_type[idx[0], idx[1]] = type_num


def generate_pos_neg_edge_type(pos_nb_nodes, neg_nb_nodes, pos_simi_edge, neg_simi_edge):
    pos_edge_type = torch.full_like(pos_simi_edge, -1)
    generate_edge_type(pos_edge_type, pos_nb_nodes, pos_simi_edge)
    neg_edge_type = torch.full_like(neg_simi_edge, -1)
    generate_edge_type(neg_edge_type, neg_nb_nodes, neg_simi_edge)

    return torch.cat((pos_edge_type, neg_edge_type), 1)


def full_empty_edge_type(edge_type_a, edge_type_b, nodes, nodes_value):
    if edge_type_a is None:
        empty_edge_type = edge_type_a
        other_edge_type = edge_type_b
    else:
        empty_edge_type = edge_type_b
        other_edge_type = edge_type_a

    indices = torch.nonzero((nodes == nodes_value), as_tuple=False)
    for idx in indices:
        if not torch.any(torch.all(torch.eq(other_edge_type, idx), dim=1)):
            empty_edge_type = torch.cat((empty_edge_type, idx.unsqueeze(0)), 0)

    return empty_edge_type


def generate_edge_type(edge_type, nodes, simi_edge):
    indices_type_1 = torch.nonzero((nodes == 1) & (simi_edge == 1), as_tuple=False)
    indices_type_2 = torch.nonzero((nodes == 1) & (simi_edge == 0), as_tuple=False)
    indices_type_3 = torch.nonzero((nodes == 0) & (simi_edge == 1), as_tuple=False)
    indices_type_4 = torch.nonzero((nodes == 0) & (simi_edge == 0), as_tuple=False)

    if indices_type_1.numel() == 0 or indices_type_2.numel() == 0:
        full_empty_edge_type(indices_type_1, indices_type_2, nodes, 1)
    if indices_type_3.numel() == 0 or indices_type_4.numel() == 0:
        print('indices_type_3 or 4 is None')
        full_empty_edge_type(indices_type_3, indices_type_4, nodes, 0)

    label_edge_type(edge_type, indices_type_1, 1)
    label_edge_type(edge_type, indices_type_2, 2)
    label_edge_type(edge_type, indices_type_3, 3)
    label_edge_type(edge_type, indices_type_4, 4)


def label_real_edge(data, arch_nodes, target_nodes):
    """
    Args:
        arch_nodes: anchor nodes
        target_nodes: target nodes
    Return:
         real_edges: indicates whether there is an edge between the anchor node and the target node in the real case, 1 means edge exists, 0 means no edge
    """
    real_edges = torch.zeros_like(target_nodes, dtype=torch.int).to(target_nodes.device)
    for i in range(arch_nodes.size(0)):
        arch_node = arch_nodes[i]
        for j in range(target_nodes.size(1)):
            if contains_edge(data, arch_node, target_nodes[i][j]):
                real_edges[i, j] = 1

    return real_edges


def classify_edge_type(data, pos_batch, neg_batch, pos_simi_edge, neg_simi_edge):
    arch_nodes = pos_batch[:, 0]
    pos_target_nodes = pos_batch[:, 1:]
    neg_target_nodes = neg_batch
    pos_nb_nodes = label_real_edge(data, arch_nodes, pos_target_nodes)
    neg_nb_nodes = label_real_edge(data, arch_nodes, neg_target_nodes)

    edge_type = generate_pos_neg_edge_type(pos_nb_nodes, neg_nb_nodes, pos_simi_edge, neg_simi_edge)

    return edge_type


# Compare the priority of the categories of i and j
# For example, if i is of type 1 edge, and j is of types 2, 3, 4 edges, then i has higher priority than j. Similarly, type 2 edges have higher priority than types 3 and 4 edges.
# Reorder t_rank based on the edge_type
def motif_t_rank_with_cf(dim_pairs, t_rank, edge_type):
    mask = edge_type[:, dim_pairs[0]] > edge_type[:, dim_pairs[1]]
    t_rank[mask] = -1

    return t_rank

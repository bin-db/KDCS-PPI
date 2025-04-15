import random
import math
import torch
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges)


# From the OGB implementation of SEAL
def do_edge_split(data, fast_split=False, val_ratio=0.2, test_ratio=0.1):
    seed = 145
    random.seed(seed)
    torch.manual_seed(seed)

    if fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        # add self-loop to train sets
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        # Sample equal number of negative and positive training examples
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def record_results(logger_file, test_r_record):
    file = open(logger_file, "a")
    file.write(f'All runs:\n')
    best_results = []
    for key, r_items in test_r_record.items():
        r_items_tensor = torch.tensor(r_items)
        max_idx = torch.argmax(r_items_tensor[:, 0])
        best_results.append([r_items_tensor[max_idx][0].item(), r_items_tensor[max_idx][1].item()])
        file.write(f'{key}:\n')
        file.write(f'auc: {r_items_tensor[max_idx][0].item()}, f1_score: {r_items_tensor[max_idx][1].item()}\n')

    file.write(f'Final results:\n')
    best_result = torch.tensor(best_results)
    auc_r = best_result[:, 0]
    f1_r = best_result[:, 1]
    print(f'auc: Test: {auc_r.mean():.4f} ± {auc_r.std():.4f}')
    file.write(f'auc Test: {auc_r.mean():.4f} ± {auc_r.std():.4f}\n')
    print(f'f1_score: Test: {f1_r.mean():.4f} ± {f1_r.std():.4f}')
    file.write(f'f1_score Test: {f1_r.mean():.4f} ± {f1_r.std():.4f}\n')
    file.write('\n')

    file.close()

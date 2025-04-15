import argparse
import itertools
import os.path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import numpy as np
from utils import do_edge_split, record_results
from models import MLP, GCN, SAGE, LinkPredictor
from os.path import exists
from torch_cluster import random_walk
from torch.nn.functional import cosine_similarity
import torch_geometric
from train_teacher_gnn import test_transductive
from dataloader import get_ppi_dataset
from select_nodes_cf import select_target_nodes, load_cf_data, load_target_nodes_data
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def cosine_loss(s, t):
    return 1 - cosine_similarity(s, t.detach(), dim=-1).mean()


def kl_loss(s, t, T):
    y_s = F.log_softmax(s / T, dim=-1)
    y_t = F.softmax(t / T, dim=-1)
    loss = F.kl_div(y_s, y_t, size_average=False) * (T ** 2) / y_s.size()[0]
    return loss


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def neighbor_samplers(row, col, sample, x, step, ps_method, ns_rate, hops):
    batch = sample

    if ps_method == 'rw':
        pos_batch = random_walk(row, col, batch, walk_length=step * hops,
                                coalesced=False)
    elif ps_method == 'nb':
        pos_batch = None
        for i in range(step):
            if pos_batch is None:
                pos_batch = random_walk(row, col, batch, walk_length=hops, coalesced=False)
            else:
                pos_batch = torch.cat(
                    (pos_batch, random_walk(row, col, batch, walk_length=hops, coalesced=False)[:, 1:]), 1)

    neg_batch = torch.randint(0, x.size(0), (batch.numel(), step * hops * ns_rate),
                              dtype=torch.long)

    return pos_batch.to("cuda"), neg_batch.to("cuda")


def train(model, predictor, t_h, teacher_predictor, data, split_edge, optimizer, args, device):
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    row, col = data.adj_t

    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)

    total_loss = total_examples = 0
    adj = np.zeros((data.x.size(0), data.x.size(0)))
    for i, j in data.edge_index.t().tolist():
        adj[i, j] = 1
        adj[j, i] = 1

    T_file_path = f'{args.cf_data_path}T_files/'
    os.makedirs(T_file_path, exist_ok=True)
    T_file = f'{T_file_path}{args.datasets}_{args.model}_T_file.pkl'
    T_f, T_cf, adj_cf, sampling_prob_mat = load_cf_data(T_file, adj, t_h.detach().cpu().numpy())

    nodes_file_path = f'{args.cf_data_path}nodes_file_path/'
    if not os.path.exists(nodes_file_path):
        os.makedirs(nodes_file_path, exist_ok=True)
    target_nodes_file = f'{nodes_file_path}{args.datasets}_{args.model}_target_nodes_file.pkl'
    pos_t_nodes, neg_t_nodes, pos_t_nodes_prob, neg_t_nodes_prob = load_target_nodes_data(target_nodes_file, data.x.size(0), adj, adj_cf, T_f,
                                                                                          T_cf, sampling_prob_mat, device,
                                                                                          args.sampling_nodes_num)

    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):
        optimizer.zero_grad()

        node_perm = next(node_loader).to(data.x.device)

        h = model(data.x)

        edge = pos_train_edge[link_perm].t()

        if args.LLP_R or args.LLP_D:
            pos_sample, neg_sample = select_target_nodes(pos_t_nodes, neg_t_nodes, pos_t_nodes_prob, neg_t_nodes_prob,
                                                         node_perm, args.select_nodes_num)

            # calculate the distribution based matching loss
            samples = torch.cat((pos_sample, neg_sample), 1)

            batch_emb = torch.reshape(h[samples[:, 0]], (samples[:, 0].size(0), 1,
                                                         h.size(1))).repeat(1, 2 * args.select_nodes_num, 1)
            t_emb = torch.reshape(t_h[samples[:, 0]], (samples[:, 0].size(0), 1,
                                                       h.size(1))).repeat(1, 2 * args.select_nodes_num, 1)

            s_r = predictor(batch_emb, h[samples[:, 1:]])
            t_r = teacher_predictor(t_emb, t_h[samples[:, 1:]])
            llp_d_loss = kl_loss(torch.reshape(s_r, (s_r.size()[0], s_r.size()[1])),
                                 torch.reshape(t_r, (t_r.size()[0], t_r.size()[1])), 1)

            sampled_nodes = [l_i for l_i in range(4)]

            dim_pairs = [x for x in itertools.combinations(sampled_nodes, r=2)]
            dim_pairs = np.array(dim_pairs).T
            teacher_rank_list = torch.zeros((len(t_r), dim_pairs.shape[1], 1)).to(t_r.device)

            mask = t_r[:, dim_pairs[0]] > (t_r[:, dim_pairs[1]] + args.margin)
            teacher_rank_list[mask] = 1
            mask2 = t_r[:, dim_pairs[0]] < (t_r[:, dim_pairs[1]] - args.margin)
            teacher_rank_list[mask2] = -1

            first_rank_list = s_r[:, dim_pairs[0]].squeeze()
            second_rank_list = s_r[:, dim_pairs[1]].squeeze()
            llp_r_loss = margin_rank_loss(first_rank_list, second_rank_list, teacher_rank_list.squeeze())

        neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0), num_neg_samples=link_perm.size(0), method='dense')

        # calculate the true_label loss
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        label_loss = bce_loss(out, train_label)

        t_out = teacher_predictor(t_h[train_edges[0]], t_h[train_edges[1]]).squeeze().detach()

        if args.LLP_D or args.LLP_R:
            loss = args.True_label * label_loss + args.KD_RM * cosine_loss(h[node_perm],
                                                                           t_h[node_perm]) + args.KD_LM * mse_loss(out,
                                                                                                                   t_out) + args.LLP_D * llp_d_loss + args.LLP_R * llp_r_loss
        else:
            loss = args.True_label * label_loss + args.KD_RM * cosine_loss(h[node_perm],
                                                                           t_h[node_perm]) + args.KD_LM * mse_loss(out,
                                                                                                                   t_out)

        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def main():
    parser = argparse.ArgumentParser(description='PPI-StudentMLP')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--t_predictor_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_output_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=512, help='Hidden layer dimension of the student MLP')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--link_batch_size', type=int, default=50000)
    parser.add_argument('--node_batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)  # default=0.005
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default='../dataset')
    parser.add_argument('--datasets', type=str, default='SHS27k')
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner', 'mlp'])
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='auc', choices=['auc', 'f1_score'], help='main evaluation metric')
    parser.add_argument('--True_label', default=1, type=float, help="true_label loss")
    parser.add_argument('--KD_RM', default=1, type=float, help="Representation-based matching KD")
    parser.add_argument('--KD_LM', default=0, type=float, help="logit-based matching KD")
    parser.add_argument('--LLP_D', default=1, type=float, help="distribution-based matching kd")
    parser.add_argument('--LLP_R', default=1, type=float, help="rank-based matching kd")
    parser.add_argument('--margin', default=0.1, type=float, help="margin for rank-based kd")
    parser.add_argument('--cf_data_path', type=str, default='../cf_data/')
    parser.add_argument('--model', type=str, default='student_mlp')
    parser.add_argument('--sampling_nodes_num', type=int, default=15)
    parser.add_argument('--select_nodes_num', type=int, default=9)
    # CodeBook setup
    parser.add_argument('--prot_num_layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=7)
    parser.add_argument('--prot_hidden_dim', type=int, default=7)
    parser.add_argument('--output_dim', type=int, default=5)
    parser.add_argument('--split_mode', type=str, default='random')
    parser.add_argument('--seed', type=int, default=256)
    parser.add_argument('--num_embeddings', type=int, default=256)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--pre_epoch', type=int, default=50)
    parser.add_argument('--mask_loss', type=float, default=0.5)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--sce_scale', type=float, default=1.5)
    parser.add_argument('--log_num', type=int, default=10)

    args = parser.parse_args()
    print(args)

    logger_file = "../results/" + args.datasets + "_KD" + ".txt"
    file = open(logger_file, "a")
    file.write(str(args) + "\n")
    if args.KD_RM != 0:
        file.write("Representation-matching\n")
    elif args.KD_LM != 0:
        file.write("Logit-matching\n")
    elif args.LLP_D != 0 or args.LLP_R != 0:
        file.write("LLP (Relational Distillation)\n")
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #  Prepare the datasets
    _, data = get_ppi_dataset(args)

    process_data_dir = '../dataset/string/processed_data/'
    if exists(process_data_dir + args.datasets + ".pkl"):
        split_edge = torch.load(process_data_dir + args.datasets + ".pkl")
    else:
        print('do_edge_split')
        split_edge = do_edge_split(data)
        torch.save(split_edge, process_data_dir + args.datasets + ".pkl")

    edge_index = split_edge['train']['edge'].t()  # size: [2, num_edges]
    data.adj_t = edge_index
    input_size = data.x.size()[1]  # dim of feats

    data.full_adj_t = data.adj_t
    data = data.to(device)

    args.node_batch_size = int(data.x.size()[0] / (split_edge['train']['edge'].size()[0] / args.link_batch_size))

    # Prepare the teacher and student model
    print('Prepare the teacher and student model')
    model = MLP(3, input_size, args.mlp_hidden_channels, args.mlp_output_channels, args.dropout).to(device)

    predictor = LinkPredictor(args.predictor, args.mlp_output_channels, args.mlp_output_channels, 1,
                              args.num_layers, args.dropout).to(device)

    pretrained_model = torch.load("../saved-models/" + args.datasets + "-" +
                                  args.encoder + ".pkl")

    teacher_predictor = LinkPredictor(args.predictor, args.t_predictor_hidden_channels,
                                      args.t_predictor_hidden_channels, 1, 2, args.dropout)
    teacher_predictor.load_state_dict(pretrained_model['predictor'], strict=True)
    teacher_predictor.to(device)

    t_h = torch.load("../saved-features/" + args.datasets + "-" + args.encoder + ".pkl")
    t_h = t_h['features']

    for para in teacher_predictor.parameters():
        para.requires_grad = False

    test_r_log = {}
    for run in range(args.runs):
        torch_geometric.seed.seed_everything(run + 1)

        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        test_r_list = []
        for epoch in range(1, 1 + args.epochs):
            train_loss = train(model, predictor, t_h, teacher_predictor, data, split_edge, optimizer, args, device)

            results, h = test_transductive(model, predictor, data, split_edge, args.link_batch_size, 'mlp',)
            test_r_list.append([results['auc'][1], results['f1_score'][1]])

            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait += 1

            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {train_loss:.4f}, '
                          f'Valid: {100 * valid_hits:.2f}%, '
                          f'Test: {100 * test_hits:.2f}%')
                print('---')

            if cnt_wait >= args.patience:
                break

        test_r_log[run] = []
        test_r_log[run].extend(test_r_list)
        
    record_results(logger_file, test_r_log)


if __name__ == '__main__':
    main()

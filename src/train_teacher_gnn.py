import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric
from utils import do_edge_split, record_results
from models import MLP, GCN, SAGE, LinkPredictor, CodeBook, GIN
from torch_geometric.nn import SAGEConv
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists
from dataloader import get_ppi_dataset, load_data, collate
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pretrain_vae(param):
    protein_data, ppi_g, ppi_list, labels = load_data(param)

    feats_dir = '../saved-features'
    output_dir = '../saved-models/vae'
    os.makedirs(feats_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, 'train_log.txt'), 'a+')

    vae_dataloader = DataLoader(protein_data, batch_size=512, shuffle=True, collate_fn=collate)
    vae_model = CodeBook(param, DataLoader(protein_data, batch_size=512, shuffle=False, collate_fn=collate)).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=float(param.learning_rate),
                                     weight_decay=float(param.weight_decay))

    for epoch in range(1, param.pre_epoch + 1):
        for iter_num, batch_graph in enumerate(vae_dataloader):

            batch_graph.to(device)

            z, e, e_q_loss, recon_loss, mask_loss = vae_model(batch_graph)
            loss_vae = e_q_loss + recon_loss + mask_loss * param.mask_loss

            vae_optimizer.zero_grad()
            loss_vae.backward()
            vae_optimizer.step()

            if (epoch - 1) % param.log_num == 0 and iter_num == 0:
                print(
                    "\033[0;30;43m Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\033[0m".format(
                        epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.write(
                    "Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\n".format(
                        epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.flush()

    torch.save(vae_model.state_dict(), os.path.join(output_dir, f'vae_{param.datasets}_model.ckpt'))

    # save protein features obtained from the encoder
    feats = vae_model.Protein_Encoder.forward(vae_model.vq_layer).to(device)
    torch.save(feats, os.path.join(feats_dir, f'vae_{param.datasets}_feats.pth'))

    del vae_model
    torch.cuda.empty_cache()


def train(model, predictor, data, split_edge, optimizer, args):
    row, col = data.adj_t
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    bce_loss = nn.BCELoss()
    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size, shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        if args.encoder == 'mlp':
            h = model(data.x)
        else:
            h = model(data.x, data.adj_t)

        neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0), num_neg_samples=perm.size(0), method='dense')

        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        loss = bce_loss(out, train_label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_transductive(model, predictor, data, split_edge, batch_size, encoder_name):
    model.eval()
    predictor.eval()

    if encoder_name == 'mlp':
        h = model(data.x)
    else:
        h = model(data.x, data.adj_t)

    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['auc'] = (roc_auc_score(valid_result.cpu().numpy(), valid_pred.cpu().numpy()),
                      roc_auc_score(test_result.cpu().numpy(), test_pred.cpu().numpy()))

    val_f1_score, thres = get_best_f1(valid_result.cpu().numpy(), valid_pred.cpu().numpy())
    test_f1_score, _ = get_best_f1(test_result.cpu().numpy(), test_pred.cpu().numpy(), thres)
    results['f1_score'] = (val_f1_score, test_f1_score)

    return results, h


def get_best_f1(labels, output, thres=0.0):
    if thres == 0.0:
        best_f1, best_thre = 0, 0
        for thres in np.linspace(0.05, 0.95, 19):
            pred = (output > thres).astype(int)
            f1 = f1_score(labels, pred, average='micro')
            if f1 > best_f1:
                best_f1 = f1
                best_thre = thres
    else:
        best_thre = thres
        pred = (output > thres).astype(int)
        best_f1 = f1_score(labels, pred, average='micro')

    return best_f1, best_thre


torch.autograd.set_detect_anomaly(True)
def main():
    parser = argparse.ArgumentParser(description='PPI-(GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='SHS27k')
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner', 'mlp'])
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='f1_score', choices=['f1_score', 'auc'],
                        help='main evaluation metric')
    # protein_encoder args
    parser.add_argument('--prot_num_layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=7)
    parser.add_argument('--ppi_hidden_dim', type=int, default=1024)
    parser.add_argument('--ppi_num_layers', type=int, default=2)
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

    vae_feats_path = f'../saved-features/vae_{args.datasets}_feats.pth'
    if not os.path.exists(vae_feats_path):
        pretrain_vae(args)

    os.makedirs("../results", exist_ok=True)
    Logger_file = "../results/" + args.datasets + "_supervised" + ".txt"
    file = open(Logger_file, "a")
    file.write(str(args))
    file.write(args.encoder + " as the encoder\n")
    file.close()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')

    prot_data, data = get_ppi_dataset(args)

    processed_data_dir = '../dataset/string/processed_data/'
    os.makedirs(processed_data_dir, exist_ok=True)
    if exists(processed_data_dir + args.datasets + ".pkl"):
        split_edge = torch.load(processed_data_dir + args.datasets + ".pkl")
    else:
        split_edge = do_edge_split(data)
        torch.save(split_edge, processed_data_dir + args.datasets + ".pkl")

    edge_index = split_edge['train']['edge'].t()
    data.adj_t = edge_index
    input_size = data.x.size()[1]
    args.metric = 'f1_score'

    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1])
    data.full_adj_t = adj_t

    data = data.to(device)

    if args.encoder == 'sage':
        model = SAGE(args.datasets, input_size, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout, SAGEConv).to(device)
    elif args.encoder == 'gcn':
        model = GCN(input_size, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder == 'gin':
        model = GIN(args).to(device)
    elif args.encoder == 'mlp':
        model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              2, args.dropout).to(device)

    val_max = 0.0
    test_r_log = {}
    for run in range(args.runs):
        torch_geometric.seed.seed_everything(run)

        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        test_r_list = []
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer, args)

            results, h = test_transductive(model, predictor, data, split_edge,
                                           args.batch_size, args.encoder, args.datasets, args)

            test_r_list.append([results['auc'][1], results['f1_score'][1]])

            if results[args.metric][0] > val_max:
                val_max = results[args.metric][0]
                if args.encoder != 'mlp':
                    os.makedirs("../saved-features", exist_ok=True)
                    os.makedirs("../saved-models", exist_ok=True)
                    torch.save({'features': h},
                               "../saved-features/" + args.datasets + "-" + args.encoder + ".pkl")
                    torch.save({'gnn': model.state_dict(), 'predictor': predictor.state_dict()},
                               "../saved-models/" + args.datasets + "-" + args.encoder + ".pkl")
            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait += 1

            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_r, test_r = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Valid: {100 * valid_r:.2f}%, '
                          f'Test: {100 * test_r:.2f}%')
                print('---')

            if cnt_wait >= args.patience:
                break

        test_r_log[run] = []
        test_r_log[run].extend(test_r_list)

    record_results(Logger_file, test_r_log)


if __name__ == "__main__":
    main()

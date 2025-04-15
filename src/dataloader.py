import os.path
import pickle
import torch
import numpy as np
import dgl
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate(samples):
    return dgl.batch_hetero(samples)


def load_data(param, skip_head=True):
    name = 0
    ppi_name = 0

    protein_name = {}
    ppi_dict = {}
    ppi_list = []
    ppi_label_list = []

    raw_data_path = '../dataset/'
    ppi_path = raw_data_path + 'string/protein.actions.{}.txt'.format(param.datasets)
    prot_seq_path = raw_data_path + 'string/protein.{}.sequences.dictionary.csv'.format(param.datasets)
    prot_r_edge_path = raw_data_path + 'string/protein.rball.edges.{}.npy'.format(param.datasets)
    prot_k_edge_path = raw_data_path + 'string/protein.knn.edges.{}.npy'.format(param.datasets)
    prot_node_path = raw_data_path + 'string/protein.nodes.{}.pt'.format(param.datasets)

    ppi_list_path = raw_data_path + 'string/{}_ppi.pkl'.format(param.datasets)
    ppi_label_list_path = raw_data_path + 'string/{}_ppi_label.pkl'.format(param.datasets)
    if os.path.exists(ppi_list_path):
        with open(ppi_list_path, 'rb') as tf:
            ppi_list = pickle.load(tf)
        with open(ppi_label_list_path, 'rb') as tf:
            ppi_label_list = pickle.load(tf)
    else:
        # get node and node name
        with open(prot_seq_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in protein_name.keys():
                    protein_name[row[0]] = name
                    name += 1

        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')
            # get edge and its label
            if line[0] < line[1]:
                temp_data = line[0] + '__' + line[1]
            else:
                temp_data = line[1] + '__' + line[0]

            if temp_data not in ppi_dict.keys():
                ppi_dict[temp_data] = ppi_name
                temp_label = 1
                ppi_label_list.append(temp_label)
                ppi_name += 1

        for ppi in tqdm(ppi_dict.keys()):
            temp = ppi.strip().split('__')
            ppi_list.append(temp)

        ppi_num = len(ppi_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = ppi_list[i][0]
            seq2_name = ppi_list[i][1]
            ppi_list[i][0] = protein_name[seq1_name]
            ppi_list[i][1] = protein_name[seq2_name]

        with open(ppi_list_path, 'wb') as tf:
            pickle.dump(ppi_list, tf)
        with open(ppi_label_list_path, 'wb') as tf:
            pickle.dump(ppi_label_list, tf)

    ppi_g = dgl.to_bidirected(dgl.graph(ppi_list))
    protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, param.datasets)

    return protein_data, ppi_g.to(device), ppi_list, torch.FloatTensor(np.array(ppi_label_list)).to(device)


class ProteinDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset):
        prot_graph_path = '../dataset/string/{}_protein_graphs.pkl'.format(dataset)
        if os.path.exists(prot_graph_path):
            with open(prot_graph_path, 'rb') as tf:
                self.prot_graph_list = pickle.load(tf)
        else:
            prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
            prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
            prot_node = torch.load(prot_node_path)

            self.prot_graph_list = []

            for i in range(len(prot_r_edge)):
                prot_seq = []
                for j in range(prot_node[i].shape[0]-1):
                    prot_seq.append((j, j+1))
                    prot_seq.append((j+1, j))

                prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid'): prot_seq,
                                          ('amino_acid', 'STR_KNN', 'amino_acid'): prot_k_edge[i],
                                          ('amino_acid', 'STR_DIS', 'amino_acid'): prot_r_edge[i]}).to(device)
                prot_g.ndata['x'] = torch.FloatTensor(prot_node[i]).to(device)

                self.prot_graph_list.append(prot_g)

            with open(prot_graph_path, 'wb') as tf:
                pickle.dump(self.prot_graph_list, tf)

    def __len__(self):
        return len(self.prot_graph_list)

    def __getitem__(self, idx):
        return self.prot_graph_list[idx]


def get_ppi_dataset(param):
    prot_data, ppi_g, _, _ = load_data(param)
    edge_index = ppi_g.edges()
    edge_index = torch.stack(edge_index)

    feats_dir = '../saved-features'
    feats = torch.load(os.path.join(feats_dir, f'vae_{param.datasets}_feats.pth'))

    dataset = Data(feats, edge_index)

    return prot_data, dataset

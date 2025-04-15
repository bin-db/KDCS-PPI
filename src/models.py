import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, GINConv
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
import dgl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GIN(torch.nn.Module):
    def __init__(self, param):
        super(GIN, self).__init__()

        self.num_layers = param.ppi_num_layers
        self.dropout = nn.Dropout(param.dropout)
        self.layers = nn.ModuleList()

        self.layers.append(GINConv(nn.Sequential(nn.Linear(param.prot_hidden_dim * 2, param.ppi_hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(param.ppi_hidden_dim, param.ppi_hidden_dim),
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(param.ppi_hidden_dim)),
                                   aggregator_type='sum',
                                   learn_eps=True))

        for _ in range(self.num_layers - 2):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(param.ppi_hidden_dim, param.ppi_hidden_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(param.ppi_hidden_dim, param.ppi_hidden_dim),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(param.ppi_hidden_dim)),
                                       aggregator_type='sum',
                                       learn_eps=True))

        self.layers.append(GINConv(nn.Sequential(nn.Linear(param.ppi_hidden_dim, param.ppi_hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(param.ppi_hidden_dim, param.hidden_channels),
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(param.hidden_channels)),
                                   aggregator_type='sum',
                                   learn_eps=True))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj_t):
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = self.dropout(x)

        x = self.layers[-1](x, adj_t)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, data_name, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, conv_layer, norm_type="none"):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        elif self.norm_type == "layer":
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(conv_layer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_channels))
        self.convs.append(conv_layer(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.norm_type != "none":
                x = self.norms[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


# link prediction via MLP or inner
class LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)

        return torch.sigmoid(x)


class GCN_Encoder(nn.Module):
    def __init__(self, param, data_loader):
        super(GCN_Encoder, self).__init__()

        self.data_loader = data_loader
        self.num_layers = param.prot_num_layers
        self.dropout = nn.Dropout(param.dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()
        # self.readout_fc = nn.Linear(param.prot_hidden_dim, param.output_dim)

        self.norms.append(nn.BatchNorm1d(param.prot_hidden_dim))
        self.fc.append(nn.Linear(param.prot_hidden_dim, param.prot_hidden_dim))
        self.layers.append(HeteroGraphConv({
            'SEQ': GraphConv(param.input_dim, param.prot_hidden_dim),
            'STR_KNN': GraphConv(param.input_dim, param.prot_hidden_dim),
            'STR_DIS': GraphConv(param.input_dim, param.prot_hidden_dim)}, aggregate='sum'))

        for i in range(self.num_layers-1):
            self.norms.append(nn.BatchNorm1d(param.prot_hidden_dim))
            self.fc.append(nn.Linear(param.prot_hidden_dim, param.prot_hidden_dim))
            self.layers.append(HeteroGraphConv({
                'SEQ': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim),
                'STR_KNN': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim),
                'STR_DIS': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim)}, aggregate='sum'))

    def forward(self, vq_layer):
        prot_embed_list = []

        for _, batch_graph in enumerate(self.data_loader):
            batch_graph.to(device)
            h = self.encoding(batch_graph)
            z, _, _ = vq_layer(h)
            batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
            prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
            prot_embed_list.append(prot_embed)

        return torch.cat(prot_embed_list, dim=0)

    def encoding(self, batch_graph):
        x = batch_graph.ndata['x']

        for i, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.norms[i](F.relu(self.fc[i](x['amino_acid'])))
            if i != self.num_layers - 1:
                x = self.dropout(x)

        return x


class GCN_Decoder(nn.Module):
    def __init__(self, param):
        super(GCN_Decoder, self).__init__()

        self.num_layers = param.prot_num_layers
        self.dropout = nn.Dropout(param.dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param.prot_hidden_dim))
            self.fc.append(nn.Linear(param.prot_hidden_dim, param.prot_hidden_dim))
            self.layers.append(HeteroGraphConv({
                'SEQ': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim),
                'STR_KNN': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim),
                'STR_DIS': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim)}, aggregate='sum'))

        self.fc.append(nn.Linear(param.prot_hidden_dim, param.input_dim))
        self.layers.append(HeteroGraphConv({
            'SEQ': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim),
            'STR_KNN': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim),
            'STR_DIS': GraphConv(param.prot_hidden_dim, param.prot_hidden_dim)}, aggregate='sum'))

    def decoding(self, batch_graph, x):
        for i, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.fc[i](x['amino_acid'])

            if i != self.num_layers - 1:
                x = self.dropout(self.norms[i](F.relu(x)))

            else:
                pass

        return x


class CodeBook(nn.Module):
    def __init__(self, param, data_loader):
        super(CodeBook, self).__init__()

        self.param = param

        self.Protein_Encoder = GCN_Encoder(param, data_loader)
        self.Protein_Decoder = GCN_Decoder(param)

        self.vq_layer = VectorQuantizer(param.prot_hidden_dim, param.num_embeddings, param.commitment_cost)

    def forward(self, batch_graph):
        z = self.Protein_Encoder.encoding(batch_graph)
        e, e_q_loss, encoding_indices = self.vq_layer(z)

        x_recon = self.Protein_Decoder.decoding(batch_graph, e)
        recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

        mask = torch.bernoulli(torch.full(size=(self.param.num_embeddings, ), fill_value=self.param.mask_ratio)).bool().to(device)
        mask_index = mask[encoding_indices]
        e[mask_index] = 0.0

        x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)

        x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
        y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
        mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param.sce_scale))

        return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, embedding_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss, encoding_indices

    def get_code_indices(self, x):
        distances = (
            torch.sum(x ** 2, dim=-1, keepdim=True) +
            torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1)**2, dim=1) -
            2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
        )

        encoding_indices = torch.argmin(distances, dim=1)

        return encoding_indices

    def quantize(self, encoding_indices):

        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)

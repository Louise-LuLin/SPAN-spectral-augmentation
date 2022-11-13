import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn import TransformerConv


def get_gnn(gnn_name, input_dim):
    if gnn_name == 'gcn':
        gconv = GCN(input_dim=input_dim, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2)
    elif gnn_name == 'gconv':
        gconv = GConv(input_dim=input_dim, hidden_dim=256, num_layers=2)
    elif gnn_name == 'sage':
        gconv = GraphSage(input_dim=input_dim, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2)
    elif gnn_name == 'sgc':
        gconv = SGC(input_dim=input_dim, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2)
    elif gnn_name == 'gat':
        gconv = GAT(input_dim=input_dim, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2)
    elif gnn_name == 'transformer':
        gconv = Transformer(input_dim=input_dim, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2)
    else:
        exit(f'unknown gnn model {gnn_name}')
        
    return gconv


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)
    
    
class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)
    

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GCN, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z
    
    
class GraphSage(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GraphSage, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class SGC(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(SGC, self).__init__()
        self.conv = SGConv(input_dim, hidden_dim, K=num_layers, cached=False)
        self.activation = activation()

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv(x, edge_index, edge_weight)
        z = self.activation(z)
        return z


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GAT, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        
        head = 8
        hidden = int(hidden_dim/8)
        self.layers.append(GATConv(input_dim, hidden, heads=head, dropout=0.6))
        for _ in range(num_layers - 1):
            intput = hidden * head
            self.layers.append(GATConv(intput, hidden_dim, heads=1, dropout=0.6))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = F.dropout(z, p=0.6, training=self.training)
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class GAT2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super().__init__()
        self.activation = activation()
        self.conv1 = GATv2Conv(input_dim, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATv2Conv(8 * 8, hidden_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.activation(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.activation(self.conv2(x, edge_index, edge_weight))
        return F.log_softmax(x, dim=-1)


class Transformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super().__init__()
        self.activation = activation()
        self.conv1 = TransformerConv(input_dim, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = TransformerConv(8 * 8, hidden_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.activation(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.activation(self.conv2(x, edge_index, edge_weight))
        return F.log_softmax(x, dim=-1) 


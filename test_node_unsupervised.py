import sys
sys.path.append('./')

import argparse
import numpy as np
import random
import os
import os.path as osp
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid

import GCL.losses as L
import GCL.augmentors as A
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast

# from GCL.data_loader import Dataset 


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = z1 + z2
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora') 
    parser.add_argument('--aug_lr', type=float, default=100, help='augmentation learning rate')
    parser.add_argument('--aug_iter', type=int, default=20, help='iteration for augmentation')
    parser.add_argument('--pf', type=float, default=0.3, help='feature probability')
    parser.add_argument('--pe', type=float, default=0.3, help='edge probability')
    parser.add_argument('--epochs', type=int, default=200, help='training epoch')
    parser.add_argument('--lr', type=float, default=0.0.001, help='edge probability')
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='cuda')
    
    
    return parser.parse_args()

def main():
    
    args = arg_parse()
    
    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1) # limit cpu use    
    
    # Load dataset
    path = osp.join(osp.expanduser('../'), 'datasets')
    dataset = Planetoid(path, name=args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    
    aug1 = A.Compose([A.EdgeFlipping(g=(data.x, data.y, data.edge_index, data.edge_attr), 
                                     nnodes=data.num_nodes, 
                                     ratio=args.pe, 
                                     lr=args.aug_lr, 
                                     iteration=args.aug_iter, 
                                     dis_type='self', 
                                     device=device, 
                                     check='no'), 
                      A.FeatureMasking(pf=args.pf)])
    aug2 = A.Compose([A.EdgeFlipping(g=(data.x, data.y, data.edge_index, data.edge_attr), 
                                     nnodes=data.num_nodes, 
                                     ratio=args.pe, 
                                     lr=args.aug_lr, 
                                     iteration=args.aug_iter, 
                                     dis_type='-self', 
                                     device=device, 
                                     check='no'), 
                      A.FeatureMasking(pf=args.pf)])

    
    gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
    gconv2 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    data = data.to(device)
    
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(1, args.epochs+1):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()

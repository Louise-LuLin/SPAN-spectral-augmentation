from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import add_edge
from GCL.augmentors.functional import get_adj_tensor, get_normalize_adj_tensor
from torch_geometric.utils import get_laplacian, to_dense_adj, dense_to_sparse
import torch
import copy

class SpectrumNoise(Augmentor):
    def __init__(self, g, pe, device):
        super(SpectrumNoise, self).__init__()
        self.pe = pe
        self.device = device
        
        x, y, edge_index, edge_weights = g
        
        # ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        # # ori_adj = to_dense_adj(edge_index)
        # ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        # # ori_e = torch.linalg.eigvalsh(ori_adj_norm)
        
        edge_index, edge_weights = get_laplacian(edge_index=edge_index, normalization='sym')
        ori_adj_norm = to_dense_adj(edge_index, edge_attr=edge_weights)
        
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        
        self.ori_adj = ori_adj_norm
        self.ori_v = ori_v.squeeze()
        self.ori_e = ori_e.squeeze()
        

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        
        aug_e = self.ori_e + self.ori_e * self.pe * (torch.randn(self.ori_e.shape).to(self.device)+1)
        aug_laplacian = self.ori_v @ torch.diag(aug_e).to(self.device)
        aug_laplacian = aug_laplacian @ self.ori_v.t()
        
        aug_adj = self.reconstruct_dense_adj_from_dense_laplacian(aug_laplacian, filter_threshold=5e-3)

        edge_index, _ = dense_to_sparse(aug_adj)
        
        # print(self.ori_adj.sum(1), aug_adj.sum(1))
        print (self.ori_adj.sum(), aug_adj.sum(), (self.ori_adj-aug_adj).abs().sum())
        
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

    
    def reconstruct_dense_adj_from_dense_laplacian(self, laplacian, filter_threshold=1e-2):
        dense_adj = laplacian.le(-filter_threshold).type(torch.float32)   
        dense_adj.fill_diagonal_(0.0)
        return dense_adj
    
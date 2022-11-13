from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import add_edge
from GCL.augmentors.functional import get_adj_tensor, get_normalize_adj_tensor
from torch_geometric.utils import get_laplacian, to_dense_adj, dense_to_sparse
import torch
import copy
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans

class ClusterFlip(Augmentor):
    def __init__(self, g, nnodes, pe, add, focus, action, supervision, device):
        super(ClusterFlip, self).__init__()
        self.pe = pe
        self.add = add
        self.focus = focus
        self.nnodes = nnodes
        self.action = action
        self.device = device
        
        x, y, edge_index, edge_weights = g
        
        if supervision == 'no':
            lap_index, lap_weights = get_laplacian(edge_index=edge_index, normalization='sym')
            lap = to_dense_adj(lap_index, edge_attr=lap_weights, max_num_nodes=y.shape[0])
            e, v = torch.symeig(lap, eigenvectors=True) # from small to large eigenvalues
            
            sc = SpectralClustering(n_clusters=7, affinity='nearest_neighbors', random_state=0)
            label = sc.fit(v.squeeze().cpu().numpy()).labels_
            label = torch.tensor(label).to(self.device)
        else:
            label = y
        
        # print (label.shape)
        # exit()
        
        self.adj_changes = self.calc_weight(label, edge_index)
        
    
    def calc_weight(self, y, edge_index):
        
        row_extend = y.repeat(y.shape[0],1).t()
        column_extend = y.repeat(y.shape[0],1)
        intra_cluster = torch.eq(row_extend, column_extend).int()  # n*n, 1 if having same labels
        inter_cluster = torch.ne(row_extend, column_extend).int()  # n*n, 1 if having diff labels
        
        adj = to_dense_adj(edge_index, max_num_nodes=self.nnodes).squeeze()  # n*n, 1 if having edge
        
        if self.action == 'remove':
            target = torch.eq(adj, 1).int()  # n*n, 1 if having edges
        else:
            target = torch.eq(adj, 0).int()  # n*n, 1 if not having edges
            
        intra_cluster_pairs = torch.logical_and(intra_cluster, target).int()
        inter_cluster_pairs = torch.logical_and(inter_cluster, target).int()
        assert torch.equal(target.int(), (intra_cluster_pairs + inter_cluster_pairs))
        
        print ('{}: total pairs={}, inter cluster={}, intra cluster={}'.format(
            self.action, target.sum(), inter_cluster_pairs.sum(), intra_cluster_pairs.sum()))
        
        if self.focus == 'intra':
            prob_intra = min(1.0, self.pe + self.add, self.pe * target.sum() / intra_cluster_pairs.sum())
            prob_inter = (self.pe * target.sum() - prob_intra * intra_cluster_pairs.sum()) / inter_cluster_pairs.sum()
        else:
            prob_inter = min(1.0, self.pe + self.add, self.pe * target.sum() / inter_cluster_pairs.sum())
            prob_intra = (self.pe * target.sum() - prob_inter * inter_cluster_pairs.sum()) / intra_cluster_pairs.sum()
        print ('focus={}, prob inter={}, prob intra={}'.format(self.focus, prob_inter, prob_intra))
        
        return intra_cluster_pairs * prob_intra + inter_cluster_pairs * prob_inter
        
        
    def random_sample(self):
        with torch.no_grad():
            s = self.adj_changes.cpu().numpy()
            sampled = np.random.binomial(1, s)
            return torch.FloatTensor(sampled).to(self.device)
    

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        
        ori_adj = to_dense_adj(edge_index, max_num_nodes=self.nnodes).squeeze()
        adj_changes = self.random_sample()
        
        if self.action == 'remove':
            modified_adj = (ori_adj - adj_changes).detach()
        else:
            modified_adj = (ori_adj + adj_changes).detach()
           
        edge_index, edge_weights = dense_to_sparse(modified_adj)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
    
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parameter import Parameter
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import get_adj_tensor, get_normalize_adj_tensor, to_dense_adj, dense_to_sparse, switch_edge
import pickle as pkl
import os

class EdgeSwitching(Augmentor):
    def __init__(self, pe1: float, pe2: float):
        super(EdgeSwitching, self).__init__()
        self.pe1 = pe1
        self.pe2 = pe2

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = switch_edge(edge_index, edge_weight=edge_weights, ratio1=self.pe1, ratio2=self.pe2)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class EdgeSwaping(Augmentor):
    def __init__(self, g, nnodes, ratio, device):
        super(EdgeSwaping, self).__init__()
        self.g = g
        self.nnodes = nnodes
        self.ratio = ratio
        self.device = device
        self.adj_changes = torch.FloatTensor(int(self.nnodes*(self.nnodes-1)/2)).fill_(self.ratio)
        
    def random_sample(self):
        with torch.no_grad():
            s = self.adj_changes.numpy()
            sampled = np.random.binomial(1, s)
            return torch.FloatTensor(sampled).to(self.device)
        
        
    def get_modified_adj(self, ori_adj, adj_changes):

        complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes
        m = m + m.t()        
        modified_adj = complementary * m + ori_adj

        return modified_adj
    

    def augment(self, g):
        x, edge_index, edge_weights = g.unfold()
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        
        adj_changes = self.random_sample()
        
        modified_adj = self.get_modified_adj(ori_adj, adj_changes).detach()
           
        edge_index, edge_weights = dense_to_sparse(modified_adj)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)



# perturb by maximizing spectral distance
class EdgeFlipping(Augmentor):
    def __init__(self, g, nnodes, ratio, lr, iteration, dis_type, device, check, save='no', sample='no'):
        super(EdgeFlipping, self).__init__()
        
        self.nnodes = nnodes
        self.ratio = ratio
        self.lr = lr
        self.iteration = iteration
        self.dis_type = dis_type
        self.device = device
        self.sample = sample
        
        self.complementary = None
        
        self.calc_weight(g, check, save)


    def get_modified_adj(self, ori_adj, adj_changes):

        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes
        m = m + m.t()        
        modified_adj = self.complementary * m + ori_adj

        return modified_adj
    
    
    def add_random_noise(self, ori_adj):
        noise = 1e-4 * torch.rand(self.nnodes, self.nnodes).to(self.device)
        return (noise + torch.transpose(noise, 0, 1))/2.0 + ori_adj

    
    def projection(self, n_perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            l = left.cpu().detach()
            r = right.cpu().detach()
            m = miu.cpu().detach()
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data-miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))
            
            
    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                b = miu
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    
    def random_sample(self):
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            # s = (s + np.transpose(s))
            if self.sample == 'yes':
                binary = np.random.binomial(1, s)
                mask = np.random.binomial(1, 0.7, s.shape)
                sampled = np.multiply(binary, mask)
            else:
                sampled = np.random.binomial(1, s)
            return torch.FloatTensor(sampled).to(self.device)
    
    
    def check_hist(self):
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            stat = {}
            stat['1.0'] = (s==1.0).sum()
            stat['(1.0,0.8)'] = (s>0.8).sum() - (s==1.0).sum()
            stat['[0.8,0.6)'] = (s>0.6).sum() - (s>0.8).sum()
            stat['[0.6,0.4)'] = (s>0.4).sum() - (s>0.6).sum()
            stat['[0.4,0.2)'] = (s>0.2).sum() - (s>0.4).sum()
            stat['[0.2,0.0]'] = (s>0.0).sum() - (s>0.2).sum()
            stat['0.0'] = (s==0.0).sum()
            print (stat)
    
    
    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is symmetric, unweighted, all-zero diagonal.
        """
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"
        
        
    def check_changes(self, ori_adj, adj_changes, y):
        m = torch.zeros((self.nnodes, self.nnodes))
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes.cpu()
        m = m + m.t()
        idx = torch.nonzero(m).numpy()
        m = m.detach().numpy()
        degree = ori_adj.sum(dim=1).cpu().numpy()
        idx2 = torch.nonzero(ori_adj.cpu()).numpy()
        
        stat = {'intra': 0, 'inter': 0, 'degree': [], 'inter_add':0, 'inter_rm': 0, 'intra_add': 0, 'intra_rm': 0, 'degree_add': [], 'degree_rm': []}
        for i in tqdm(idx):
            d = degree[i[0]] + degree[i[1]]
            if ori_adj[i[0], i[1]] == 1:  # rm
                if y[i[0]] == y[i[1]]:  # intra
                    stat['intra_rm'] += m[i[0], i[1]]
                if y[i[0]] != y[i[1]]:  # inter
                    stat['inter_rm'] += m[i[0], i[1]]
                stat['degree_rm'].append(d/2)
            if ori_adj[i[0], i[1]] == 0:  # add
                if y[i[0]] == y[i[1]]:  # intra
                    stat['intra_add'] += m[i[0], i[1]]
                if y[i[0]] != y[i[1]]:  # inter
                    stat['inter_add'] += m[i[0], i[1]]
                stat['degree_add'].append(d/2)
        for i in tqdm(idx2):
            d = degree[i[0]] + degree[i[1]]
            if y[i[0]] == y[i[1]]:  # intra
                stat['intra'] += 1
            if y[i[0]] != y[i[1]]:  # inter
                stat['inter'] += 1
            stat['degree'].append(d/2)
                
        stat['degree_rm'] = sum(stat['degree_rm'])/(len(stat['degree_rm'])+0.1)
        stat['degree_add'] = sum(stat['degree_add'])/(len(stat['degree_add'])+0.1)
        stat['degree'] = sum(stat['degree'])/(len(stat['degree'])+0.1)
        
        print(stat)

#     def calc_weight_random(self, g, check, save='no'):
#         self.adj_changes = Parameter(torch.FloatTensor(int(self.nnodes*(self.nnodes-1)/2)), requires_grad=True).to(self.device)
#         torch.nn.init.uniform_(self.adj_changes, 0.0, 0.001)
        
#         x, y, edge_index, edge_weights = g
        
#         x = x.to(self.device)
#         ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
#         # ori_adj = to_dense_adj(edge_index)
#         ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
#         # ori_e = torch.linalg.eigvalsh(ori_adj_norm)
#         ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        
#         n_perturbations = int(self.ratio * (ori_adj.sum()/2))
        
#         spectral_perb = torch.normal(0, 1, ori_e.shape) * 
#         self.adj_changes = 
#         print (torch.sum(self.adj_changes))
#         self.projection(n_perturbations)
#         print (torch.sum(self.adj_changes))
        

    def calc_weight(self, g, check, save='no'):
        self.adj_changes = Parameter(torch.FloatTensor(int(self.nnodes*(self.nnodes-1)/2)), requires_grad=True).to(self.device)
        torch.nn.init.uniform_(self.adj_changes, 0.0, 0.001)
        
        x, y, edge_index, edge_weights = g
        
        x = x.to(self.device)
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        # ori_adj = to_dense_adj(edge_index)
        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        # ori_e = torch.linalg.eigvalsh(ori_adj_norm)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        eigen_norm = torch.norm(ori_e)
        
        n_perturbations = int(self.ratio * (ori_adj.sum()/2))
        with tqdm(total=self.iteration, desc='Spectral Augment') as pbar:
            verb = max(1, int(self.iteration/10))
            for t in range(1, self.iteration+1):
                modified_adj = self.get_modified_adj(ori_adj, self.adj_changes)
                
                # add noise to make the graph asymmetric
                modified_adj_noise = modified_adj
                modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = get_normalize_adj_tensor(modified_adj_noise, device=self.device)
                # e = torch.linalg.eigvalsh(adj_norm_noise)
                e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
                eigen_self = torch.norm(e)
                
                # spectral distance
                eigen_mse = torch.norm(ori_e-e)
                
                if self.dis_type == 'l2':
                    reg_loss = eigen_mse / eigen_norm
                elif self.dis_type == '-l2':
                    reg_loss = -eigen_mse / eigen_norm
                elif self.dis_type == 'self':
                    reg_loss = eigen_self / eigen_norm
                    
                    # n = 100
                    # idx = torch.argsort(e)[:n]
                    # mask = torch.zeros_like(e).bool()
                    # mask[idx] = True
                    # eigen_low = torch.norm(e*mask, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # idx2 = torch.argsort(e, descending=True)[:n]
                    # mask2 = torch.zeros_like(e).bool()
                    # mask2[idx2] = True
                    # eigen_high = torch.norm(e*mask2, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # reg_loss = eigen_low - eigen_high
                    
                elif self.dis_type == '-self':
                    reg_loss = -eigen_self / eigen_norm
                    
                    # n = 100
                    # idx = torch.argsort(e)[:n]
                    # mask = torch.zeros_like(e).bool()
                    # mask[idx] = True
                    # eigen_low = torch.norm(e*mask, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # idx2 = torch.argsort(e, descending=True)[:n]
                    # mask2 = torch.zeros_like(e).bool()
                    # mask2[idx2] = True
                    # eigen_high = torch.norm(e*mask2, p=2)
                    # # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    
                    # reg_loss = - eigen_low + eigen_high
                    
                elif self.dis_type.startswith('low'):
                    # low-rank loss in GF-attack
                    n = int(self.dis_type.replace('low',''))
                    idx = torch.argsort(e)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = eigen_gf
                elif self.dis_type.startswith('-low'):
                    # low-rank loss in GF-attack
                    n = int(self.dis_type.replace('low',''))
                    idx = torch.argsort(e)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = -eigen_gf
                elif self.dis_type.startswith('high'):
                    # high-rank loss in GF-attack
                    n = int(self.dis_type.replace('high',''))
                    idx = torch.argsort(e, descending=True)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = eigen_gf
                elif self.dis_type.startswith('-high'):
                    # high-rank loss in GF-attack
                    n = int(self.dis_type.replace('high',''))
                    idx = torch.argsort(e, descending=True)[:n]
                    mask = torch.zeros_like(e).bool()
                    mask[idx] = True
                    eigen_gf = torch.norm(e*mask, p=2)
                    # eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, x), p=2), 2)
                    reg_loss = -eigen_gf
                elif self.dis_type.startswith('sep'):
                    
                    # n = int(self.dis_type.replace('sep',''))
                    # mask = torch.zeros_like(e).bool()
                    # mask[-n:] = True
                    # eigen_high = torch.masked_select(e, mask)
                    # reg_loss = eigen_high @ eigen_high.t() / (e @ e.t())
                    
                    # mask = e.ge(0.0)  # [-1, 1]
                    # eigen_high = torch.masked_select(e, mask)
                    # reg_loss = eigen_high @ eigen_high.t() / (e @ e.t())
                    
                    mask = e.le(0.0)  # [-1, 1]
                    ori_high = torch.masked_select(ori_e, mask)
                    high = torch.masked_select(e, mask)
                    mask2 = e.ge(0.0)
                    ori_low = torch.masked_select(ori_e, mask2)
                    low = torch.masked_select(e, mask2)
                    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    # reg_loss = cos(ori_high, high) + 1 - cos(ori_low, low)
                    reg_loss = cos(ori_high, high)
                    # reg_loss = torch.norm(ori_high - high, p=2)**2 / torch.norm(ori_e - e, p=2)**2
                else:
                    exit(f'unknown distance metric: {self.dis_type}')
                
                self.loss = reg_loss
                
                adj_grad = torch.autograd.grad(self.loss, self.adj_changes)[0]

                lr = self.lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)
                
                before_p = torch.clamp(self.adj_changes, 0, 1).sum()
                before_l = self.adj_changes.min()
                before_r = self.adj_changes.max()
                before_m = torch.clamp(self.adj_changes, 0, 1).sum()/torch.count_nonzero(self.adj_changes)
                self.projection(n_perturbations)
                after_p = self.adj_changes.sum()
                after_l = self.adj_changes.min()
                after_r = self.adj_changes.max()
                after_m = self.adj_changes.sum()/torch.count_nonzero(self.adj_changes)
                
                if t%verb == 0:
                    print (
                        '-- Epoch {}, '.format(t), 
                        'reg loss = {:.4f} | '.format(reg_loss),
                        'ptb budget/b/a = {:.1f}/{:.1f}/{:.1f}'.format(n_perturbations, before_p, after_p),
                        'min b/a = {:.4f}/{:.4f}'.format(before_l, after_l),
                        'max b/a = {:.4f}/{:.4f}'.format(before_r, after_r),
                        'mean b/a = {:.4f}/{:.4f}'.format(before_m, after_m))
                    self.check_hist()

                pbar.set_postfix({'reg_loss': reg_loss.item(), 'before_p': before_p.item(), 'after_p': after_p.item()})
                pbar.update()
    
        if check == 'yes':
            self.check_changes(ori_adj, self.adj_changes, y)
            
        if save == 'yes':
            out_dir = '../check'
            os.makedirs(out_dir, exist_ok=True)
            
            output_path = os.path.join(out_dir, self.dis_type+'_'+str(self.ratio)+'_'+str(self.lr)+'_'+str(self.iteration)+'.bin')
            res = {'ori_e': ori_e, 'e': e, 'adj_change': self.adj_changes.detach().cpu(), 'ori_adj': ori_adj.detach().cpu()}
            with open(output_path, 'wb') as file:
                pkl.dump(res, file)
            # exit()
    
    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        
        adj_changes = self.random_sample()
        # adj_changes = self.adj_changes
        
        modified_adj = self.get_modified_adj(ori_adj, adj_changes).detach()
        self.check_adj_tensor(modified_adj)
           
        edge_index, edge_weights = dense_to_sparse(modified_adj)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
    
    
    def augment2(self, g: Graph) -> Graph:
        self.adj_changes = Parameter(torch.FloatTensor(int(self.nnodes*(self.nnodes-1)/2)), requires_grad=True).to(self.device)
        torch.nn.init.uniform_(self.adj_changes, 0.0, 0.001)
        
        x, edge_index, edge_weights = g.unfold()
        
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        # ori_adj = to_dense_adj(edge_index)
        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        # ori_e = torch.linalg.eigvalsh(ori_adj_norm)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        eigen_norm = torch.norm(ori_e)
        
        # print(ori_adj.shape, ori_adj_norm.shape)
        # exit('')
        
        n_perturbations = int(self.ratio * (ori_adj.sum()/2))
        with tqdm(total=self.iteration, desc='Spectral Augment') as pbar:
            for t in range(1, self.iteration+1):
                modified_adj = self.get_modified_adj(ori_adj, self.adj_changes)
                
                # add noise to make the graph asymmetric
                modified_adj_noise = modified_adj
                modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = get_normalize_adj_tensor(modified_adj_noise, device=self.device)
                # e = torch.linalg.eigvalsh(adj_norm_noise)
                e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
                eigen_self = torch.norm(e)
                
                # spectral distance
                eigen_mse = torch.norm(ori_e-e)
                
                if self.dis_type == 'l2':
                    reg_loss = eigen_mse / eigen_norm
                elif self.dis_type == 'normDiv':
                    reg_loss = eigen_self / eigen_norm
                else:
                    exit(f'unknown distance metric: {self.dis_type}')
                
                # if t%10 == 0:
                #     print ('-- Epoch {}, '.format(t), 
                #         'ptb budget/true = {:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()),
                #         'reg loss = {:.4f} | '.format(reg_loss),
                #         'eigen_norm = {:4f} | '.format(eigen_norm),
                #         'eigen_mse = {:.4f} | '.format(eigen_mse),
                #         'eigen_self = {:.4f} | '.format(eigen_self))

                self.loss = reg_loss
                
                adj_grad = torch.autograd.grad(self.loss, self.adj_changes)[0]

                lr = self.lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)
                    
                # if t%10 == 0:
                #     print('before proj: budget/true={:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()))
                
                before_p = torch.clamp(self.adj_changes, 0, 1).sum()
                self.projection(n_perturbations)
                after_p = torch.clamp(self.adj_changes, 0, 1).sum()
                
                # if t%10 == 0:
                #     print('after proj: budget/true={:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()))
                    
                pbar.set_postfix({'reg_loss': reg_loss.item(), 'eigen_mse': eigen_mse.item(), 'before_p': before_p.item(), 'after_p': after_p.item()})
                pbar.update()
        
        adj_changes = self.random_sample()
        
        # print("final: ptb budget/true= {:.1f}/{:.1f}".format(n_perturbations, self.adj_changes.sum()))
        modified_adj = self.get_modified_adj(ori_adj, adj_changes).detach()
        self.check_adj_tensor(modified_adj)
            
        edge_index, edge_weights = dense_to_sparse(modified_adj)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

    
    
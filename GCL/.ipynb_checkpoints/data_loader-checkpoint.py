import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.datasets import LastFMAsia
from torch_geometric.datasets import GitHub
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import WikiCS
from torch_geometric.datasets import AmazonProducts
from torch_geometric.datasets import Yelp
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

class Dataset(Data):
    def __init__(self, data_path, data_name):
        super().__init__()

        if data_name in ['Cora', 'Citeseer', 'PubMed']:
            # Cora: node_num=2708, edge_num=5278, feature_num=1433, class_num=7
            # Citeseer: node_num=3327, edge_num=4552, feature_num=3703, class_num=6
            # PubMed: node_num=19717, edge_num=44324, feature_num=500, class_num=3
            dataset = Planetoid(root=data_path, name=data_name, transform=T.NormalizeFeatures()) # can use transform=T.NormalizeFeatures()
        elif data_name == 'FacebookPagePage': 
            # node_num=22470, edge_num=171002, feature_num=128, class_num=4
            dataset = FacebookPagePage(root=data_path+'/'+data_name)
        elif data_name == 'LastFMAsia':  
            # node_num=7624, edge_num=27806, feature_num=128, class_num=18
            dataset = LastFMAsia(root=data_path+'/'+data_name)
        elif data_name == 'GitHub':  
            # node_num=37700, edge_num=289003, feature_num=128, class_num=2
            dataset = GitHub(root=data_path+'/'+data_name)
        elif data_name == 'Actor':  
            # node_num=7600, edge_num=15009, feature_num=932, class_num=5
            dataset = Actor(root=data_path+'/'+data_name)
        elif data_name in ['Cornell', 'Texas', 'Wisconsin']:
            dataset = WebKB(root=data_path, name=data_name)
        elif data_name == 'WikiCS':  
            # node_num=11701, edge_num=148555, feature_num=300, class_num=10
            dataset = WikiCS(root=data_path+'/'+data_name)
        elif data_name == 'AmazonProducts':  
            # OOM issue: node_num=1569960, edge_num=132169734, feature_num=200, class_num=107
            dataset = AmazonProducts(root=data_path+'/'+data_name)
        elif data_name == 'Yelp':  
            # OOM issue: node_num=716847, edge_num=6977409, feature_num=300, class_num=100
            dataset = Yelp(root=data_path+'/'+data_name)
        elif data_name == 'Flickr':  
            # node_num=89250, edge_num=449878, feature_num=500, class_num=7
            dataset = Flickr(root=data_path+'/'+data_name)
        elif data_name == 'Reddit':  
            # OOM Issue: node_num=232965, edge_num=57307946, feature_num=602, class_num=41
            dataset = Reddit(root=data_path+'/'+data_name)
        elif data_name == 'Reddit2':  
            # OOM Issue: node_num=232965, edge_num=11606919, feature_num=602, class_num=41
            dataset = Reddit2(root=data_path+'/'+data_name)
        elif data_name in ['Computers', 'Photo']: 
            # Computers: node_num=13752, edge_num=245861, feature_num=767, class_num=10
            # Photo: node_num=7650, edge_num=119081, feature_num=745, class_num=8
            dataset = Amazon(root=data_path, name=data_name)
        elif data_name in ['CS', 'Physics']: 
            # CS: node_num=18333, edge_num=81894, feature_num=6805, class_num=15
            # Physics: node_num=34493, edge_num=247962, feature_num=8415, class_num=5
            dataset = Coauthor(root=data_path, name=data_name)
        elif data_name in ['Cora_Full', 'Cora_ML', 'Citeseer_Full', 'DBLP']: 
            # Cora_Full: node_num=19793, edge_num=63421, feature_num=8710, class_num=70
            # Cora_ML: node_num=2995, edge_num=8158, feature_num=2879, class_num=7
            # Citeseer_Full: node_num=4230, edge_num=5337, feature_num=602, class_num=6
            # DBLP: node_num=17716, edge_num=52867, feature_num=1639, class_num=4
            if data_name == 'Cora_Full':
                data_name = 'Cora'
            if data_name == 'Citeseer_Full':
                data_name = 'Citeseer'
            dataset = CitationFull(root=data_path, name=data_name)
        else:
            exit('unknown dataset: {}'.format(data_name))            

        # train_ratio = 0.5
        # val_ratio = 0.2
        # test_ratio = 0.3
        # data, train_mask, val_mask, test_mask = self.split(train_ratio, val_ratio, test_ratio)
        
        self.data = dataset[0]
        
        # we mainly use data.x, data.edge_index, data.edge_attr
        print('=== Data statistics ===')
        log  = 'data.x: {}, {}\ndata.y: {}, {}\ndata.edge_index: {}, {}\ndata.edge_attr: {}'
        print(log.format(type(self.data.x), self.data.x.shape, type(self.data.y), self.data.y.shape, 
                         type(self.data.edge_index), self.data.edge_index.shape, type(self.data.edge_attr)))

        self.num_classes = dataset.num_classes

        log = 'Node num: {}\nNode feature size: {}\nNode class num: {}\nEdge num: {}'
        print(log.format(self.data.num_nodes, self.data.num_node_features, self.num_classes, int(self.data.num_edges/2)))
        log = 'Train:val:test = {}:{}:{}'
        print(log.format(torch.sum(self.data.train_mask), torch.sum(self.data.val_mask), torch.sum(self.data.test_mask)))


    def split(self, train_ratio, val_ratio, test_ratio):
        train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)

        train_mask.fill_(False)
        for c in range(self.num_classes):
            idx = (self.data.y == c).nonzero(as_tuple=False).view(-1)
            num_train_per_class = int(idx.size(0) * train_ratio)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            train_mask[idx] = True

        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        val_mask.fill_(False)
        num_val = int(remaining.size(0) * (val_ratio/(val_ratio + test_ratio)))
        val_mask[remaining[:num_val]] = True

        test_mask.fill_(False)
        num_test = int(remaining.size(0) * (test_ratio/(val_ratio + test_ratio)))
        test_mask[remaining[num_val:num_val + num_test]] = True

        return train_mask, val_mask, test_mask

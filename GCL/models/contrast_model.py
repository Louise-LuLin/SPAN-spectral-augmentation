import torch

from GCL.losses import Loss
from GCL.models import get_sampler, link_sampler


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask


class SingleBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(SingleBranchContrast, self).__init__()
        assert mode == 'G2L'  # only global-local pairs allowed in single-branch contrastive learning
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h, g, batch=None, hn=None, extra_pos_mask=None, extra_neg_mask=None):
        if batch is None:  # for single-graph datasets
            assert hn is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, neg_sample=hn)
        else:  # for multi-graph datasets
            assert batch is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, batch=batch)

        pos_mask, neg_mask = add_extra_mask(pos_mask, neg_mask, extra_pos_mask, extra_neg_mask)
        loss = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask, **self.kwargs)
        return loss


class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5

class LinkPredictionContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(LinkPredictionContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = link_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1, h2, h3, edge_index):
        anchor12, sample12, pos_mask12, neg_mask12 = self.sampler(anchor=h1, sample=h2, edge_index=edge_index)
        loss12 = self.loss(anchor=anchor12, sample=sample12, pos_mask=pos_mask12, neg_mask=neg_mask12, **self.kwargs)
        
        anchor13, sample13, pos_mask13, neg_mask13 = self.sampler(anchor=h1, sample=h3, edge_index=edge_index)
        loss13 = self.loss(anchor=anchor13, sample=sample13, pos_mask=pos_mask13, neg_mask=neg_mask13, **self.kwargs)
        
        anchor23, sample23, pos_mask23, neg_mask23 = self.sampler(anchor=h2, sample=h3, edge_index=edge_index)
        loss23 = self.loss(anchor=anchor23, sample=sample23, pos_mask=pos_mask23, neg_mask=neg_mask23, **self.kwargs)
        
        return loss12+loss13+loss23
    
class AdvBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, reg_weight: float = 1.0, **kwargs):
        super(AdvBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.reg_weight = reg_weight
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None and h3 is not None
            ##### <adv,a> #####
            anchor12, sample12, pos_mask12, neg_mask12 = self.sampler(anchor=h1, sample=h2)
            anchor21, sample21, pos_mask21, neg_mask21 = self.sampler(anchor=h2, sample=h1)
            
            pos_mask12, neg_mask12 = add_extra_mask(pos_mask12, neg_mask12, extra_pos_mask, extra_neg_mask)
            pos_mask21, neg_mask21 = add_extra_mask(pos_mask21, neg_mask21, extra_pos_mask, extra_neg_mask)
            l12 = self.loss(anchor=anchor12, sample=sample12, pos_mask=pos_mask12, neg_mask=neg_mask12, **self.kwargs)
            l21 = self.loss(anchor=anchor21, sample=sample21, pos_mask=pos_mask21, neg_mask=neg_mask21, **self.kwargs)
            
            dis1 = (l12 + l21) * 0.5
            
            ##### <adv,b> #####
            anchor13, sample13, pos_mask13, neg_mask13 = self.sampler(anchor=h1, sample=h3)
            anchor31, sample31, pos_mask31, neg_mask31 = self.sampler(anchor=h3, sample=h1)
            
            pos_mask13, neg_mask13 = add_extra_mask(pos_mask13, neg_mask13, extra_pos_mask, extra_neg_mask)
            pos_mask31, neg_mask31 = add_extra_mask(pos_mask31, neg_mask31, extra_pos_mask, extra_neg_mask)
            l13 = self.loss(anchor=anchor13, sample=sample13, pos_mask=pos_mask13, neg_mask=neg_mask13, **self.kwargs)
            l31 = self.loss(anchor=anchor31, sample=sample31, pos_mask=pos_mask31, neg_mask=neg_mask31, **self.kwargs)
            
            dis2 = (l13 + l31) * 0.5
            
            ##### <a,b> #####
            anchor32, sample32, pos_mask32, neg_mask32 = self.sampler(anchor=h3, sample=h2)
            anchor23, sample23, pos_mask23, neg_mask23 = self.sampler(anchor=h2, sample=h3)
            
            pos_mask32, neg_mask32 = add_extra_mask(pos_mask32, neg_mask32, extra_pos_mask, extra_neg_mask)
            pos_mask23, neg_mask23 = add_extra_mask(pos_mask23, neg_mask23, extra_pos_mask, extra_neg_mask)
            l32 = self.loss(anchor=anchor32, sample=sample32, pos_mask=pos_mask32, neg_mask=neg_mask32, **self.kwargs)
            l23 = self.loss(anchor=anchor23, sample=sample23, pos_mask=pos_mask23, neg_mask=neg_mask23, **self.kwargs)
            
            dis3 = (l32 + l23) * 0.5
            
            # s(a1,b1)>s(a1,adv1)
            l231_pos = self.loss(anchor=anchor23, sample=sample23, pos_mask=pos_mask23, sample2=sample21, **self.kwargs) # h2 h3 h1
            l321_pos = self.loss(anchor=anchor32, sample=sample32, pos_mask=pos_mask32, sample2=sample31, **self.kwargs) # h3 h2 h1
            dis_pos = (l231_pos + l321_pos) * 0.5
            # s(a1,adv2)>s(a1,b2)
            l213_neg = self.loss(anchor=anchor21, sample=sample21, pos_mask=neg_mask21, sample2=sample23, **self.kwargs) # h2 h1 h3
            l312_neg = self.loss(anchor=anchor31, sample=sample31, pos_mask=neg_mask31, sample2=sample32, **self.kwargs) # h3 h1 h2
            dis_neg = (l213_neg + l312_neg) * 0.5
        
            return dis1 + dis2 + dis3 + self.reg_weight * (dis_pos + dis_neg)
            
            
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5


class AdvBranchContrast2(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, 
                 reg1: float = 1.0, reg2: float = 1.0, reg3: float = 1.0, reg4: float = 1.0, **kwargs):
        super(AdvBranchContrast2, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.reg1 = reg1
        self.reg2 = reg2
        self.reg3 = reg3
        self.reg4 = reg4
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None and h3 is not None and h4 is not None
            ##### <adv1,adv2> #####
            anchor12, sample12, pos_mask12, neg_mask12 = self.sampler(anchor=h1, sample=h2)
            anchor21, sample21, pos_mask21, neg_mask21 = self.sampler(anchor=h2, sample=h1)
            
            pos_mask12, neg_mask12 = add_extra_mask(pos_mask12, neg_mask12, extra_pos_mask, extra_neg_mask)
            pos_mask21, neg_mask21 = add_extra_mask(pos_mask21, neg_mask21, extra_pos_mask, extra_neg_mask)
            l12 = self.loss(anchor=anchor12, sample=sample12, pos_mask=pos_mask12, neg_mask=neg_mask12, **self.kwargs)
            l21 = self.loss(anchor=anchor21, sample=sample21, pos_mask=pos_mask21, neg_mask=neg_mask21, **self.kwargs)
            
            dis1 = (l12 + l21) * 0.5
            
            ##### <a,b> #####
            anchor34, sample34, pos_mask34, neg_mask34 = self.sampler(anchor=h3, sample=h4)
            anchor43, sample43, pos_mask43, neg_mask43 = self.sampler(anchor=h4, sample=h3)
            
            pos_mask34, neg_mask34 = add_extra_mask(pos_mask34, neg_mask34, extra_pos_mask, extra_neg_mask)
            pos_mask43, neg_mask43 = add_extra_mask(pos_mask43, neg_mask43, extra_pos_mask, extra_neg_mask)
            l34 = self.loss(anchor=anchor34, sample=sample34, pos_mask=pos_mask34, neg_mask=neg_mask34, **self.kwargs)
            l43 = self.loss(anchor=anchor43, sample=sample43, pos_mask=pos_mask43, neg_mask=neg_mask43, **self.kwargs)
            
            dis2 = (l34 + l43) * 0.5
            
            # s(a1,b1)>s(adv1,adv2)
            l231_pos = self.loss(anchor=anchor34, sample=sample34, pos_mask=pos_mask34, sample2=sample12, anchor2=anchor12, **self.kwargs) # h2 h3 h1
            l321_pos = self.loss(anchor=anchor43, sample=sample43, pos_mask=pos_mask43, sample2=sample21, anchor2=anchor21, **self.kwargs) # h3 h2 h1
            dis_pos = (l231_pos + l321_pos) * 0.5
            # s(a1,adv2)>s(a1,b2)
            l213_neg = self.loss(anchor=anchor12, sample=sample12, pos_mask=neg_mask34, sample2=sample34, **self.kwargs) # h2 h1 h3
            l312_neg = self.loss(anchor=anchor21, sample=sample21, pos_mask=neg_mask43, sample2=sample43, **self.kwargs) # h3 h1 h2
            dis_neg = (l213_neg + l312_neg) * 0.5
        
            return self.reg1*dis1 + self.reg2*dis2 + self.reg3*dis_pos + self.reg4*dis_neg
            
            
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5


class BootstrapContrast(torch.nn.Module):
    def __init__(self, loss, mode='L2L'):
        super(BootstrapContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=False)

    def forward(self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None,
                g1_pred=None, g2_pred=None, g1_target=None, g2_target=None,
                batch=None, extra_pos_mask=None):
        if self.mode == 'L2L':
            assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=h1_target, sample=h2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=h2_target, sample=h1_pred)
        elif self.mode == 'G2G':
            assert all(v is not None for v in [g1_pred, g2_pred, g1_target, g2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=g2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=g1_pred)
        else:
            assert all(v is not None for v in [h1_pred, h2_pred, g1_target, g2_target])
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                pos_mask1 = pos_mask2 = torch.ones([1, h1_pred.shape[0]], device=h1_pred.device)
                anchor1, sample1 = g1_target, h2_pred
                anchor2, sample2 = g2_target, h1_pred
            else:
                anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=h2_pred, batch=batch)
                anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=h1_pred, batch=batch)

        pos_mask1, _ = add_extra_mask(pos_mask1, extra_pos_mask=extra_pos_mask)
        pos_mask2, _ = add_extra_mask(pos_mask2, extra_pos_mask=extra_pos_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2)

        return (l1 + l2) * 0.5


class WithinEmbedContrast(torch.nn.Module):
    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1, h2):
        l1 = self.loss(anchor=h1, sample=h2, **self.kwargs)
        l2 = self.loss(anchor=h2, sample=h1, **self.kwargs)
        return (l1 + l2) * 0.5

from matplotlib.pyplot import sca
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math
import sys

from abc import ABC, abstractmethod


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class GetSubnet_Str(autograd.Function):
    @staticmethod
    def forward(ctx, weights, k, dim=0):
        # Get the subnetwork by sorting the scores and using the top k%
        weights = torch.norm(weights, p=1, dim=dim)

        out = weights.clone()
        _, idx = weights.flatten().sort()
        j = int((1 - k) * weights.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None



class Operation_ST(ABC):
    def __init__(self):
        self.use_subset = True

        self.ste_sigmoid = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
    
    def enable_ste_sigmoid(self):
        self.ste_sigmoid = True

    def disable_score_grad(self):
        # del self.scores
        self.scores.grad = None
        self.scores.requires_grad = False

    def init_weight_with_score(self, use_subnet=True):
        self.weight.data = self.weight.data * GetSubnet.apply(self.clamped_scores, self.prune_rate).detach()
        self.use_subset = use_subnet

    def init_score_with_weight_mag(self):
        self.scores.data = GetSubnet.apply(self.weight.data.abs(), self.prune_rate).data

    def init_score_with_negative_weight_mag(self):
        self.scores.data = GetSubnet.apply(-self.weight.data.abs(), self.prune_rate).data


    def init_score_with_weight_rank(self):
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # print('score before:', self.scores.flatten()[:10])

        score_flatten = self.scores.data.flatten()
        weight_mag_flatten = self.weight.data.abs().flatten()

        _, ind_weight_mag = weight_mag_flatten.sort()
        _, ind_score_mag = score_flatten.abs().sort()

        score_flatten[ind_weight_mag] = score_flatten[ind_score_mag]  # assign the elements with the largest weight magnitude with the score with largest magnitude

        # print('weight:', self.weight.data.abs().flatten()[:10])
        # print('score after:', self.scores.flatten()[:10])
        # input()


    def init_score_with_weight_mag_with_scale(self, scale):
        # print('weight mean:', self.weight.data.mean())
        
        self.scores.data = self.weight.data / scale


    def init_score(self, init_method='kaiming_uniform', scale=1):
        if init_method == 'kaiming_uniform' or init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.scores)
        elif init_method == 'xavier_normal':
            nn.init.xavier_uniform_(self.scores)
        elif init_method == 'uniform':
            nn.init.uniform_(self.scores)
        elif init_method == 'normal':
            nn.init.normal_(self.scores)
        elif init_method == 'zero':
            nn.init.constant_(self.scores, 0)
        elif init_method == 'one':
            nn.init.constant_(self.scores, 1)
        elif init_method == 'weight_magnitude':
            self.init_score_with_weight_mag()
        elif init_method == 'negative_weight_magnitude':
            self.init_score_with_negative_weight_mag()
        elif init_method == 'weight_rank':
            self.init_score_with_weight_rank()
        elif init_method == 'weight_magnitude_with_scale':
            self.init_score_with_weight_mag_with_scale(scale=scale)
        else:
            print('No such init method:', init_method)
            sys.exit()


    def init_score_with_xavier_normal(self):
        nn.init.xavier_uniform_(self.scores)

    def init_score_with_uniform(self):
        nn.init.uniform_(self.scores)


    @property
    def clamped_scores(self):
        return self.scores.abs()

    @property
    def masked_weight(self):
        if self.use_subset:
            subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * subnet
        else:
            w = self.weight
        
        return w

    def get_subnet(self):
        return GetSubnet.apply(self.clamped_scores, self.prune_rate).detach()

    def forward(self, x):
        if self.use_subset:
            if self.ste_sigmoid:
                subnet = (GetSubnet.apply(self.clamped_scores, self.prune_rate) - torch.sigmoid(self.clamped_scores)).detach() + torch.sigmoid(self.clamped_scores)
            else:
                subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * subnet
        else:
            w = self.weight

        x = self._forward(x, w)

        return x


    @abstractmethod
    def _forward(self, x, w):
        pass



# Not learning weights, finding subnet
class Conv2d_ST(Operation_ST, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        Operation_ST.__init__(self)
        nn.Conv2d.__init__(self, *args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def _forward(self, x, w):
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)



# Not learning weights, finding subnet
class Conv1d_ST(Operation_ST, nn.Conv1d):
    def __init__(self, *args, **kwargs):
        Operation_ST.__init__(self)
        nn.Conv1d.__init__(self, *args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def _forward(self, x, w):
        return F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)



# Not learning weights, finding subnet
class Linear_ST(Operation_ST, nn.Linear):
    def __init__(self, *args, **kwargs):
        Operation_ST.__init__(self)
        nn.Linear.__init__(self, *args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def _forward(self, x, w):
        return F.linear(x, w, self.bias)


class Linear_ST_OutStr(Operation_ST, nn.Linear):
    def __init__(self, *args, **kwargs):
        Operation_ST.__init__(self)
        nn.Linear.__init__(self, *args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def _forward(self, x, w):
        return F.linear(x, w, self.bias)

    def init_score_with_weight(self):
        self.scores.data = GetSubnet_Str.apply(self.weight.data.abs(), self.prune_rate, 1).data.view(-1,1)


class Linear_ST_InStr(Operation_ST, nn.Linear):
    def __init__(self, *args, **kwargs):
        Operation_ST.__init__(self)
        nn.Linear.__init__(self, *args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(1, self.weight.size()[1]))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def _forward(self, x, w):
        return F.linear(x, w, self.bias)

    def init_score_with_weight(self):
        self.scores.data = GetSubnet_Str.apply(self.weight.data.abs(), self.prune_rate, 0).data.view(1,-1)


import torch
import torch.nn as nn
import torch.nn.functional as F

class XeLayer(nn.Module):
    def __init__(self, node_num, in_feature, out_feature, graph_num):
        super(XeLayer, self).__init__()
        self.in_feature = in_feature;
        self.out_feature = out_feature;
        self.graph_num = graph_num
        self.node_num = node_num;
        self.hid = out_feature/graph_num;
        self.bypass = nn.Linear(self.in_feature, self.out_feature);
        self.linear = nn.Linear(self.in_feature, self.out_feature);
        #self.linears = nn.ModuleList([nn.Linear(self.in_hid, self.out_hid) for i in range(graph_num)]);
        
        
    def forward(self, x, L):
        '''
        input shape: batch_size * n * in_feature;
        output shape: batch_size * n * out_feature;
        '''
        batch_size = x.size(0);
        
        xx = self.linear(x);
        h = []
        xx = torch.split(xx, self.hid, 2);
        for i in range(self.graph_num):
            hh = torch.bmm(L[i], xx[i]).view(batch_size * self.node_num, self.hid)
            h += [hh];
        
        h = torch.cat(h, 1);
            
        x = x.view(batch_size * self.node_num, self.in_feature);
        x = self.bypass(x);
        x = x + h
        res = x.view(batch_size, self.node_num, self.out_feature);
        return res;
        
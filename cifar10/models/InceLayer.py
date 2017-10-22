import torch
import torch.nn as nn
import torch.nn.functional as F

class InceLayer(nn.Module):
    def __init__(self, node_num, in_feature, out_feature, graph_num, depth):
        super(InceLayer, self).__init__()
        self.in_feature = in_feature;
        self.out_feature = out_feature / 2;
        self.in_hid = in_feature / graph_num;
        
        self.graph_num = graph_num
        self.node_num = node_num;
        self.depth = depth;
        self.bypass = []
        for i in range(self.depth):
            self.bypass += [nn.Linear(self.in_feature, self.in_feature)];
        self.bypass = nn.ModuleList(self.bypass);
        self.linear = nn.Linear(self.in_feature, self.out_feature);
        
        
    def forward(self, x, L):
        '''
        input shape: batch_size * n * in_feature;
        output shape: batch_size * n * out_feature;
        '''
        batch_size = x.size(0);
        
        for i in range(self.depth):
            h = [];
            x = x.view(batch_size, self.node_num, self.in_feature);
            xx = torch.split(x, self.in_hid, 2);
            for j in range(self.graph_num):
                hh = torch.bmm(L[j][i], xx[j]).view(batch_size * self.node_num, self.in_hid)
                h += [hh];
            h = torch.cat(h, 1);
            x = x.view(batch_size * self.node_num, self.in_feature);
            x = self.bypass[i](x);
            x = x + h
        
        x = self.linear(x);
        res = x.view(batch_size, self.node_num, self.out_feature);
        return res;
        
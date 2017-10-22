import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from XeLayer import XeLayer 


class GraphConv(nn.Module):
    def __init__(self, node_num, in_feature, out_feature, loc_feature, graph_num):
        super(GraphConv, self).__init__()
        self.node_num = node_num;
        self.in_feature = in_feature;
        self.out_feature = out_feature;
        self.loc_feature = loc_feature;
        self.graph_num = graph_num
        self.Lhid = 128;
        self.fc1 = nn.ModuleList([nn.Linear(loc_feature, self.Lhid) for i in range(graph_num)]);
        self.fc2 = nn.ModuleList([nn.Linear(self.Lhid, 1) for i in range(graph_num)]);
        self.xe = XeLayer(self.node_num, self.in_feature, self.out_feature, self.graph_num);
        
    def forward(self, x, maps, L_idx):
        m = self.node_num
        nn_num = L_idx.size(0)/m;
        batch_size = x.size(0);
        maps = maps.view(-1, self.loc_feature);
        L_list = [];
        for i in range(self.graph_num):
            context = F.tanh(self.fc1[i](maps))
            context = self.fc2[i](context);
            context = context.view(m, nn_num);
            attn = F.softmax(context);
            attn = attn.view(m * nn_num);
            L = Variable(torch.zeros(m * m).cuda());
            L[L_idx] = attn;
            L = L.view(m, m);
            L = L.expand(batch_size, *L.size());
            L_list += [L];
        x = self.xe(x, L_list);
        
        return x;
        
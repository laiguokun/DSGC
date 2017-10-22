import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PoolingLayer import Pooling

# hyper-parameters for the subsampled setting

cfgs = [256, 256, 512, 512, 512, 512, 1024, 1024]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3];
nodes_num_list = [256, 256, 64, 64, 16, 16, 4, 4];
node_num_graph = [256, 64, 16, 4];
nn_num_list = [16, 16, 12, 12, 8, 8, 3, 3];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3]
pool_num = 4;

# hyper-parameters for the origin setting

'''
cfgs = [256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
nodes_num_list = [1024, 1024, 256, 256, 64, 64, 16, 16, 4, 4];
nn_num_list = [8, 8, 8, 8, 8, 8, 8, 8, 3, 3];
node_num_graph = [1024, 256, 64, 16, 4];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3, -1, 4]
pool_num = 5;
'''

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = data.w
        x = self.w
        self.linears, self.linears2, self.linears3, self.linears4, self.batch_norm = [], [], [], [], [];
        for i in range(len(cfgs)):
            self.linears += [nn.Linear(x, cfgs[i])];
            self.linears2 += [nn.Linear(x, cfgs[i])]
            self.linears3 += [nn.Linear(x, cfgs[i])]
            self.linears4 += [nn.Linear(x, cfgs[i])]
            self.batch_norm += [nn.BatchNorm1d(cfgs[i])];
            x = cfgs[i]
            
        self.linears = nn.ModuleList(self.linears);
        self.linears2 = nn.ModuleList(self.linears2);
        self.linears3 = nn.ModuleList(self.linears3);
        self.linears4 = nn.ModuleList(self.linears4);
        self.batch_norm = nn.ModuleList(self.batch_norm)
        self.output = None;
        
        pooling_layers = [];
        for i in range(pool_num):
            pooling_layers += [Pooling(data.pooling_ks[i])];
        self.pooling_layers = nn.ModuleList(pooling_layers);
        self.index_list = data.id_list
        
        self.fc1 = nn.Linear(cfgs[-1], 512);
        self.fc2 = nn.Linear(512, 10);
        self.dropout = nn.Dropout(args.dropout);
        self.L = data.L_list
        self.L2 = []
        for j in range(pool_num):
            self.L2.append(2 * torch.mm(self.L[j], self.L[j]) - Variable(torch.eye(node_num_graph[j]).cuda()));
        self.L3 = []
        for j in range(pool_num):
            self.L3.append(2 * torch.mm(self.L[j], self.L2[j]) - self.L[j]);
        
    
    def linear_layer(self, i, x, L, L2, L3, batch_size):
        in_feature = x.size(2);
        Lx = torch.bmm(L,x);
        m = nodes_num_list[i];
        Lx = Lx.view(batch_size * m, in_feature);
        Lx = self.linears[i](Lx);
        L2x = torch.bmm(L2,x);
        L2x = L2x.view(batch_size * m, in_feature);
        L2x = self.linears3[i](L2x);
        L3x = torch.bmm(L3,x);
        L3x = L3x.view(batch_size * m, in_feature);
        L3x = self.linears4[i](L3x);        
        x = x.view(batch_size * m, in_feature);
        x = self.linears2[i](x);
        x = x + Lx + L3x + L2x;
        x = x.view(batch_size, m, -1);
        return x;
        
    def forward(self, inputs):
        x = inputs[0];L = inputs[1];
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        L , L2, L3 = [], [], []
        for i in range(pool_num):
            L.append(self.L[i].expand(batch_size, *self.L[i].size()));
            L2.append(self.L2[i].expand(batch_size, *self.L2[i].size()));
            L3.append(self.L3[i].expand(batch_size, *self.L3[i].size()));
        graph_id = 0;
        for i in range(len(cfgs)):
            x = self.linear_layer(i, x, L[graph_id], L2[graph_id], L3[graph_id], batch_size);
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);
            if (pool_list[i] != -1):
                graph_id += 1;
                pool_id = pool_list[i]
                x = self.pooling_layers[pool_id](x, self.index_list[pool_id]);
                x = self.dropout(x);
        x = x.view(batch_size, cfgs[-1]);
        x = F.relu(self.fc1(x))
        x = self.dropout(x);
        x = self.fc2(x)
        return x
    
        
        
        
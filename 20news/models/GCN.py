import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from PoolingLayer import Pooling


cfgs = [64] * 5
L_cfgs = [0] * 5;
nodes_num_list = [1000] * 5;
nn_num_list = [32] * 5;
pool_list = [-1] * 5
last_layer = 1000;

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = data.w
        
        x = self.w;
        self.linears = []
        self.linears2 = []
        self.batch_norm = []
        for i in range(len(cfgs)):
            self.linears += [nn.Linear(x, cfgs[i])];
            self.linears2 += [nn.Linear(x, cfgs[i])]
            self.batch_norm += [nn.BatchNorm1d(cfgs[i])];
            x = cfgs[i];
        self.linears = nn.ModuleList(self.linears);
        self.linears2 = nn.ModuleList(self.linears2);
        self.batch_norm = nn.ModuleList(self.batch_norm);
        self.dropout = nn.Dropout(args.dropout);
        self.output = None;
        
        self.fc1 = nn.Linear(last_layer * cfgs[-1], 512)
        self.fc2 = nn.Linear(512, 20)
        
        self.dropout = nn.Dropout(args.dropout);
    
    def linear_layer(self, i, x, L, batch_size):
        in_feature = x.size(2);
        L = L.expand(batch_size, *L.size());
        Lx = torch.bmm(L,x);
        Lx = Lx.view(batch_size * nodes_num_list[i], in_feature);
        Lx = self.linears[i](Lx);
        x = x.view(batch_size * nodes_num_list[i], in_feature);
        x = self.linears2[i](x);
        x = x + Lx;
        x = x.view(batch_size, nodes_num_list[i], -1);
        #x = self.dropout(x);
        return x;
        
    def forward(self, inputs):
        x = inputs[0];L = inputs[1];
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
            
        for i in range(len(cfgs)):
            x = self.linear_layer(i, x, L, batch_size);
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);
            x = self.dropout(x);
        x = x.view(batch_size, last_layer * cfgs[-1]);
        x = F.relu(self.fc1(x))
        x = self.dropout(x);
        x = self.fc2(x)
        return x
    
    
        
        
        
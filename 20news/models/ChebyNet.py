import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
        
        
        self.fc1 = nn.Linear(cfgs[-1] * 1000, 512);
        self.fc2 = nn.Linear(512, 20);
        self.dropout = nn.Dropout(args.dropout);
        
    
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
        L2 = 2 * torch.mm(L, L) - Variable(torch.eye(1000).cuda())
        L3 = 2 * torch.mm(L, L2) - L
        L = (L.expand(batch_size, *L.size()));
        L2 = (L2.expand(batch_size, *L2.size()));
        L3 = (L3.expand(batch_size, *L3.size()));
        for i in range(len(cfgs)):
            x = self.linear_layer(i, x, L, L2, L3, batch_size);
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);
            x = self.dropout(x);
        x = x.view(batch_size, 1000 * cfgs[-1]);
        x = F.relu(self.fc1(x))
        x = self.dropout(x);
        x = self.fc2(x)
        return x
    
        
        
        
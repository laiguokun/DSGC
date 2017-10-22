import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

'''
cfgs = [256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
nodes_num_list = [1024, 1024, 256, 256, 64, 64, 16, 16, 4, 4];
nn_num_list = [8, 8, 8, 8, 8, 8, 8, 8, 3, 3];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3, -1, 4]
pool_num = 5;
graph_num = [8] * 10;
'''

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = data.w 
        self.hid = 32
        self.linears = [nn.Linear(self.w, self.hid) for i in range(self.num_layers+1)];
        self.linears = nn.ModuleList(self.linears);
        self.fc1 = nn.Linear(self.hid * (self.num_layers + 1) * self.m, 256);
        self.fc2 = nn.Linear(256, 20);
        self.dropout = nn.Dropout(args.dropout);
        self.output = None;
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(self.hid) for i in range(self.num_layers+1)]);
        
        self.dropout = nn.Dropout(args.dropout);
        
    def forward(self, inputs):
        x = inputs[0];L = inputs[1];
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        
        t = x.view(batch_size * self.m, self.w);
        s = [F.relu(self.linears[0](t))];
        
        P = L
        
        for i in range(1, self.num_layers + 1):
            PP = P.expand(batch_size, *P.size());
            t = torch.bmm(PP, x);
            t = t.view(batch_size * self.m, self.w);
            t = self.linears[i](t);
            t = t.view(batch_size, self.m, self.hid);
            t = t.permute(0,2,1).contiguous();
            t = self.batch_norm[i](t);
            t = t.permute(0,2,1).contiguous();
            t = t.view(batch_size * self.m, self.hid)
            s += [F.relu(t)]
            P = torch.mm(L, P);
        s = torch.cat(s, 1);
        s = s.view(batch_size, self.m * self.hid * (self.num_layers + 1));
        s = self.dropout(s);
        s = F.relu(self.fc1(s));
        s = self.dropout(s);
        s = self.fc2(s);
        return s
    
        
        
        

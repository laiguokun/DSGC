import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from PoolingLayer import Pooling

'''
#cfgs = [256, 256, 512, 512, 512, 512, 1024, 1024]
cfgs = [128, 128, 256, 256, 256, 256, 512, 512]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3];
nodes_num_list = [256, 256, 64, 64, 16, 16, 4, 4];
nn_num_list = [16, 16, 12, 12, 8, 8, 3, 3];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3]
pool_num = 4;
J_num = [16] * 8;
'''

#cfgs = [256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024]
cfgs = [128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
nodes_num_list = [1024, 1024, 256, 256, 64, 64, 16, 16, 4, 4];
nn_num_list = [8, 8, 8, 8, 8, 8, 8, 8, 3, 3];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3, -1, 4]
pool_num = 5;
J_num = [16] * 10;

last_layer = 1;

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
        self.index_list = data.id_list
        self.sigma = [];
        self.mu = [];
        for i in range(len(cfgs)):
            self.linears2 += [nn.Linear(x, cfgs[i])];
            self.batch_norm += [nn.BatchNorm1d(cfgs[i])];
            m = nodes_num_list[i];
            self.sigma += [[]];
            self.mu += [[]]
            for j in range(J_num[i]):
                self.linears += [nn.Linear(x, cfgs[i])];
                self.sigma[i] += [Parameter(torch.zeros(5,).cuda())];
                self.mu[i] += [Parameter(torch.zeros(5,).cuda())];
            x = cfgs[i];
            
        self.linears = nn.ModuleList(self.linears);
        self.linears2 = nn.ModuleList(self.linears2);
        self.batch_norm = nn.ModuleList(self.batch_norm);
        self.dropout = nn.Dropout(args.dropout);
        self.output = None;
        
        pooling_layers = [];
        for i in range(pool_num):
            pooling_layers += [Pooling(data.pooling_ks[i])];
        self.pooling_layers = nn.ModuleList(pooling_layers);
        
        self.fc1 = nn.Linear(last_layer * cfgs[-1], 512)
        self.fc2 = nn.Linear(512, 10)
        
        self.dropout = nn.Dropout(args.dropout);
        self.reset_parameters();

    def reset_parameters(self):
        for i in range(len(self.sigma)):
            for j in range(J_num[i]):
                self.sigma[i][j].data.normal_(0, 1)
                self.mu[i][j].data.normal_(0, 1);
    
    def linear_layer(self, i, x, embed, idx, batch_size, offset):
        batch_size = x.size(0);
        embed = embed.view(-1,5);
        edge_num = embed.size(0);
        m = nodes_num_list[i];
        nn_num = nn_num_list[i];
        
        in_feature = x.size(2);
        
        s = Variable(torch.zeros(batch_size * m, cfgs[i]).cuda());
        
        for j in range(J_num[i]):
            mu = self.mu[i][j].expand(edge_num, 5);
            sigma = torch.diag(self.sigma[i][j]);
            u = embed - mu;
            tmp = torch.mm(u, sigma);
            tmp = tmp.view(edge_num, 1, 5);
            u = u.view(edge_num, 5, 1);
            tmp = torch.bmm(tmp, u);
            tmp = tmp.view(edge_num,);
            w = torch.exp(tmp * -0.5);
            L = Variable(torch.zeros(m * m).cuda());
            attn = w;
            attn = attn.view(m,nn_num);
            attn = F.softmax(attn);
            attn = attn.view(m * nn_num);
            L[idx] = attn;
            L = L.view(m, m);       
            L = L.expand(batch_size, *L.size());
            Lx = torch.bmm(L,x);
            Lx = Lx.view(batch_size * m, in_feature);
            Lx = self.linears[j+offset](Lx);
            s += Lx;

        x = x.view(batch_size * m, in_feature);
        x = self.linears2[i](x);
        x = x + s;
        x = x.view(batch_size, m, -1);
        return x;
        
        
    def forward(self, inputs):
        x = inputs[0]; maps = inputs[1]; L_idx = inputs[2]; 
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        offset = 0;
        for i in range(len(cfgs)):
            x = self.linear_layer(i, x, maps[L_cfgs[i]], L_idx[L_cfgs[i]], batch_size, offset);
            offset += J_num[i];
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);
            if (pool_list[i] != -1):
                pool_id = pool_list[i]
                x = self.pooling_layers[pool_id](x, self.index_list[pool_id]);
                x = self.dropout(x);
        x = x.view(batch_size, last_layer * cfgs[-1]);
        x = F.relu(self.fc1(x))
        x = self.dropout(x);
        x = self.fc2(x)
        return x
    
    
        
        
        
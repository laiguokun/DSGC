import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from PoolingLayer import Pooling


cfgs = [64] * 5
L_cfgs = [0] * 5;
nodes_num_list = [1000] * 5;
nn_num_list = [32] * 5;
J_num = [32] * 5;
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
        self.sigma = [];
        self.mu = [];
        self.embed_dim = data.embed_dim
        
        self.encode_dim = 5;
        self.edge_fc = nn.Linear(self.embed_dim, self.encode_dim)
        
        for i in range(len(cfgs)):
            self.linears += [nn.Linear(x, cfgs[i])];
            self.linears2 += [nn.Linear(x, cfgs[i])];
            self.batch_norm += [nn.BatchNorm1d(cfgs[i])];
            m = nodes_num_list[i];
            self.sigma += [[]];
            self.mu += [[]]
            for j in range(J_num[i]):
                self.sigma[i] += [Parameter(torch.zeros(self.encode_dim,).cuda())];
                self.mu[i] += [Parameter(torch.zeros(self.encode_dim,).cuda())];
            x = cfgs[i];
            
        
        self.linears = nn.ModuleList(self.linears);
        self.linears2 = nn.ModuleList(self.linears2);
        self.batch_norm = nn.ModuleList(self.batch_norm);
        self.dropout = nn.Dropout(args.dropout);
        self.output = None;
        
        self.fc1 = nn.Linear(last_layer * cfgs[-1], 512)
        self.fc2 = nn.Linear(512, 20)
        
        self.dropout = nn.Dropout(args.dropout);
        self.reset_parameters();

    def reset_parameters(self):
        for i in range(len(self.sigma)):
            for j in range(J_num[i]):
                self.sigma[i][j].data.normal_(0, 1)
                self.mu[i][j].data.normal_(0, 1);
    
    def linear_layer(self, i, x, L, batch_size):
        in_feature = x.size(2);
        L = L.expand(batch_size, *L.size());
        Lx = torch.bmm(L,x);
        Lx = Lx.view(batch_size * nodes_num_list[i], in_feature);
        Lx = self.linears[i](Lx);
        #x = Lx.view(batch_size, nodes_num_list[i], -1);
        x = x.view(batch_size * nodes_num_list[i], in_feature);
        x = self.linears2[i](x);
        x = x + Lx;
        x = x.view(batch_size, nodes_num_list[i], -1);
        return x;
        
    def get_L(self, i, embed, idx):
        batch_size = embed.size(0);
        embed = embed.view(-1, self.embed_dim);
        embed = self.edge_fc(embed);
        edge_num = embed.size(0);
        m = nodes_num_list[i];
        nn_num = nn_num_list[i];
        w = Variable(torch.zeros((edge_num, )).cuda());
        for j in range(J_num[i]):
            mu = self.mu[i][j].expand(edge_num, self.encode_dim);
            sigma = torch.diag(self.sigma[i][j]);
            u = embed - mu;
            tmp = torch.mm(u, sigma);
            tmp = tmp.view(edge_num, 1, self.encode_dim);
            u = u.view(edge_num, self.encode_dim, 1);
            tmp = torch.bmm(tmp, u);
            tmp = tmp.view(edge_num, );
            w += torch.exp(tmp * -0.5);
        context = w.view(m, nn_num);
        attn = F.softmax(context);
        attn = attn.view(m * nn_num);
        L = Variable(torch.zeros(m * m).cuda());
        L[idx] = attn;
        L = L.view(m, m);
        return L;
        
        
    def forward(self, inputs):
        x = inputs[0]; maps = inputs[1]; L_idx = inputs[2]; 
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
            
        for i in range(len(cfgs)):
            L = self.get_L(i, maps, L_idx);
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
    
    
        
        
        
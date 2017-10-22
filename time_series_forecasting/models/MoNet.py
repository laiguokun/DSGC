import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


cfgs = [64] * 10 
J_num = [8] * 10;

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = data.P
        self.nn_num = data.nn_num
        if (args.mask):
            self.w *= 2;
        self.num_layers = args.num_layers;
        
        x = self.w;
        self.linears = []
        self.linears2 = []
        self.batch_norm = []
        self.sigma = [];
        self.mu = [];
        for i in range(len(cfgs)):
            self.linears2 += [nn.Linear(x, cfgs[i])];
            self.batch_norm += [nn.BatchNorm1d(cfgs[i])];
            m = self.m
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
        
        self.fc1 = nn.Linear(cfgs[-1], 512)
        self.fc2 = nn.Linear(512, 1)
        
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
        m = self.m;
        nn_num = self.nn_num;
        
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
            x = self.linear_layer(i, x, maps, L_idx, batch_size, offset);
            offset += J_num[i];
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);
            x = self.dropout(x);
        x = x.view(batch_size * self.m, cfgs[-1]);
        x = F.relu(self.fc1(x))
        x = self.dropout(x);
        x = self.fc2(x)
        x = x.view(batch_size, self.m);
        return x
    
    
        
        
        
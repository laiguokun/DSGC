import torch
import torch.nn as nn
import torch.nn.functional as F
from XeLayer import XeLayer
from torch.nn.parameter import Parameter
from torch.autograd import Variable

_INF = float('inf');

cfgs = [64] * 10 
graph_num = [4] * 10;

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.w = data.P;
        if (args.mask):
            self.w *= 2;
        self.num_layers = args.num_layers;
        self.embed = args.embed;
        self.sigma = args.sigma;
        self.m = data.m;
        self.nn_num = data.nn_num;
        self.node2vec = None;
        if (self.embed):
            self.node2vec = nn.Embedding(self.m, self.embed);
            self.input = self.w + self.embed;
        else:
            self.input = self.w;
        
        x = self.input * graph_num[0];
        layers = []
        norm_layers = []
        for i in range(len(cfgs)):
            layers += [XeLayer(self.m, x, cfgs[i], graph_num[i])];
            norm_layers += [nn.BatchNorm1d(cfgs[i])];
            x = cfgs[i];
        
        self.fc1 = nn.Linear(cfgs[-1], 512);
        self.fc2 = nn.Linear(512, 1);
        
        self.linears = nn.ModuleList(layers);
        self.batch_norm = nn.ModuleList(norm_layers);
        
        self.dropout = nn.Dropout(args.dropout);
        
        self.Lhid = 256
        
        self.L_linear1 = nn.ModuleList([nn.Linear(5, self.Lhid) for i in range(sum(graph_num))]);
        self.L_linear2 = nn.ModuleList([nn.Linear(self.Lhid, 1) for i in range(sum(graph_num))]);
    
    def reset_parameters(self):
        self.self_embed.data.normal_(0, 1)
    
    def get_L(self, embed, idx, batch_size, layer, graph_id):
        m = self.m;
        nn_num = self.nn_num;
        embed = embed.view(-1,5);
        embed = F.tanh(self.L_linear1[layer](embed))
        context = self.L_linear2[layer](embed);
        context = context.view(m, nn_num);
        attn = F.softmax(context);
        attn = attn.view(m * nn_num);
        L = Variable(torch.zeros(m * m).cuda());
        L[idx] = attn;
        L = L.view(m, m);
        L = L.expand(batch_size, *L.size());
        return L
    
    def forward(self, inputs):
        x = inputs[0]; maps = inputs[1]; L_idx = inputs[2]; 
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        embed = None;
        if (self.embed):
            idx = inputs[3];
            embed = self.node2vec(idx);
            embed = embed.expand(batch_size, *embed.size());
            x = torch.cat((x,embed),2);
        
        
        L = []
        s = 0;
        for i in range(len(cfgs)):
            L += [[]];
            for j in range(graph_num[i]):
                L[i].append(self.get_L(maps, L_idx, batch_size, s + j, i));
            s = s + graph_num[i];
            
        x = torch.cat([x for i in range(graph_num[0])],2);    
        for i in range(len(cfgs)):
            x = self.linears[i](x, L[i]);
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
    
        
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from XeLayer import XeLayer
from PoolingLayer import Pooling
from torch.nn.parameter import Parameter
from torch.autograd import Variable

cfgs = [64] * 5
L_cfgs = [0] * 5;
nodes_num_list = [1000] * 5;
nn_num_list = [64] * 5;
pool_list = [-1] * 5
graph_num = [1] * 5;
last_layer = 1000;

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.w = data.w
        self.embed = args.embed;
        self.embed_dim = data.embed_dim
        if (args.sparse):
            self.index_list = data.id_list;
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
            layers += [XeLayer(nodes_num_list[i], x, cfgs[i], graph_num[i])];
            norm_layers += [nn.BatchNorm1d(cfgs[i])];
            x = cfgs[i];
        
        pooling_layers = [];
        for i in range(3):
            pooling_layers += [Pooling(data.pooling_ks[i])];
        self.pooling_layers = nn.ModuleList(pooling_layers);
        
        self.fc1 = nn.Linear(cfgs[-1] * last_layer, 512);
        self.fc2 = nn.Linear(512, 20);
        
        self.linears = nn.ModuleList(layers);
        self.batch_norm = nn.ModuleList(norm_layers);
        
        self.dropout = nn.Dropout(args.dropout);
        
        self.Lhid = 256
        
        self.L_linear1 = nn.ModuleList([nn.Linear(self.embed_dim, self.Lhid) for i in range(sum(graph_num))]);
        self.L_linear2 = nn.ModuleList([nn.Linear(self.Lhid, 1) for i in range(sum(graph_num))]);
    
    def reset_parameters(self):
        self.self_embed.data.normal_(0, 1)
    
    def get_L(self, embed, idx, batch_size, layer, graph_id):

        m = nodes_num_list[graph_id];
        nn_num = nn_num_list[graph_id];
        embed = embed.view(-1,self.embed_dim);
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
        x = inputs[0]; maps_list = inputs[1]; L_list = inputs[2]; id_list = inputs[3];
        batch_size = x.size(0);
        embed = None;
        if (self.embed):
            idx = inputs[3];
            embed = self.node2vec(idx);
            embed = embed.expand(batch_size, *embed.size());
        x = x.transpose(2,1).contiguous();
        
        
        L = []
        s = 0;
        for i in range(len(cfgs)):
            L += [[]];
            for j in range(graph_num[i]):
                L[i].append(self.get_L(maps_list[L_cfgs[i]], L_list[L_cfgs[i]], batch_size, s + j, i));
            s = s + graph_num[i];
            
        x = torch.cat([x for i in range(graph_num[0])],2);    
        for i in range(len(cfgs)):
            x = self.linears[i](x, L[i]);
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
    
        
        
        
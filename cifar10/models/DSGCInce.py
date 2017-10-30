import torch
import torch.nn as nn
import torch.nn.functional as F
from InceLayer import InceLayer
from PoolingLayer import Pooling
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# hyper-parameters for the subsampled setting

cfgs = [256, 256, 512, 512, 512, 512, 1024, 1024]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3];
nodes_num_list = [256, 256, 64, 64, 16, 16, 4, 4];
nn_num_list = [16, 16, 12, 12, 8, 8, 3, 3];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3]
pool_num = 4;
graph_num = [8] * 8;

# hyper-parameters for the origin setting

'''
cfgs = [256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
#cfgs = [256, 256, 256, 256, 256, 256, 256, 256]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
nodes_num_list = [1024, 1024, 256, 256, 64, 64, 16, 16, 4, 4];
nn_num_list = [8, 8, 8, 8, 8, 8, 8, 8, 3, 3];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3, -1, 4]
pool_num = 5;
graph_num = [4] * 10;
'''
last_layer = 1;
class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.w = data.w
        if (args.sparse):
            self.index_list = data.id_list;
        
        self.input = self.w
        
        x = self.input * graph_num[0];
        layers = []
        norm_layers = []
        for i in range(len(cfgs)):
            layers += [InceLayer(nodes_num_list[i], x, cfgs[i], graph_num[i], 1),
                       InceLayer(nodes_num_list[i], x, cfgs[i], graph_num[i], 2)];
            norm_layers += [nn.BatchNorm1d(cfgs[i])];
            x = cfgs[i];
        
        pooling_layers = [];
        for i in range(pool_num):
            pooling_layers += [Pooling(data.pooling_ks[i])];
        self.pooling_layers = nn.ModuleList(pooling_layers);
        
        self.fc1 = nn.Linear(cfgs[-1] * last_layer, 512);
        self.fc2 = nn.Linear(512, 10);
        
        self.linears = nn.ModuleList(layers);
        self.batch_norm = nn.ModuleList(norm_layers);
        
        self.dropout = nn.Dropout(args.dropout);
        
        self.Lhid = 128
        
        self.L_linear1 = nn.ModuleList([nn.Linear(5, self.Lhid) for i in range(sum(graph_num) * 3)]);
        self.L_linear2 = nn.ModuleList([nn.Linear(self.Lhid, 1) for i in range(sum(graph_num) * 3)]);
    
    def reset_parameters(self):
        self.self_embed.data.normal_(0, 1)
    
    def get_L(self, embed, idx, batch_size, layer, graph_id):
        
        m = nodes_num_list[graph_id];
        nn_num = nn_num_list[graph_id];
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
        x = inputs[0]; maps_list = inputs[1]; L_list = inputs[2]; id_list = inputs[3];
        batch_size = x.size(0);
        
        x = x.transpose(2,1).contiguous();
        
        
        L1 = []
        L2 = []
        s = 0;
        for i in range(len(cfgs)):
            L1 += [[]];
            L2 += [[]];
            for j in range(graph_num[i]):
                index = (s + j) * 3;
                L1[i] += [[self.get_L(maps_list[L_cfgs[i]], L_list[L_cfgs[i]], batch_size, index, i)]];
                L2[i] += [[self.get_L(maps_list[L_cfgs[i]], L_list[L_cfgs[i]], batch_size, index + 1, i),
                             self.get_L(maps_list[L_cfgs[i]], L_list[L_cfgs[i]], batch_size, index + 2, i)]];
            s = s + graph_num[i];
            
        x = torch.cat([x for i in range(graph_num[0])],2);    
        for i in range(len(cfgs)):
            x1 = self.linears[i * 2](x, L1[i]);
            x2 = self.linears[i * 2 + 1](x, L2[i]);
            x = torch.cat((x1,x2), 2);
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
    
        
        
        
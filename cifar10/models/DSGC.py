import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphConv import GraphConv
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
cfgs = [256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024]
#cfgs = [256, 256, 256, 256, 256, 256, 256, 256]
L_cfgs = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
nodes_num_list = [1024, 1024, 256, 256, 64, 64, 16, 16, 4, 4];
nn_num_list = [8, 8, 8, 8, 8, 8, 8, 8, 3, 3];
pool_list = [-1, 0, -1, 1, -1, 2, -1, 3, -1, 4]
pool_num = 5;
graph_num = [8] * 10;
'''
last_layer = 1;

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.w = data.w
        if (args.sparse):
            self.index_list = data.id_list;
        
        x = self.w;
        layers = []
        norm_layers = []
        for i in range(len(cfgs)):
            layers += [GraphConv(nodes_num_list[i], x, cfgs[i], 5, graph_num[i])];
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
    
    def reset_parameters(self):
        self.self_embed.data.normal_(0, 1)
    
    def forward(self, inputs):
        x = inputs[0]; maps_list = inputs[1]; L_list = inputs[2]; 
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
          
        for i in range(len(cfgs)):
            x = self.linears[i](x, maps_list[L_cfgs[i]], L_list[L_cfgs[i]]);
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
    
        
        
        
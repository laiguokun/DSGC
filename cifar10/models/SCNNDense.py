import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphConv import GraphConv
from PoolingLayer import Pooling
from torch.nn.parameter import Parameter
from torch.autograd import Variable

#hyper-parameters for the origin setting
'''
nodes_num_list = [1024, 256, 64, 16, 4];
blocks = [6, 12, 24, 16]
pool_num = 5;
'''

#hyper-parameters for the subsampled setting

nodes_num_list = [256, 64, 16, 4];
pool_num = 4;
graph_num = 4;
blocks = [12, 24, 16]


growth_rate = 32

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, graph_id):
        super(Bottleneck, self).__init__()
        self.graph_id = graph_id;
        self.node_num = nodes_num_list[graph_id];
        self.bn = nn.BatchNorm1d(in_planes)
        self.conv = GraphConv(self.node_num, in_planes, growth_rate, 5, graph_num)
        self.dropout = nn.Dropout(0.2);

    def forward(self, x, maps, L_idx):
        out = x;
        out = out.permute(0,2,1).contiguous();
        out = self.bn(out);
        out = out.permute(0,2,1).contiguous();
        out = F.relu(out);
        out = self.dropout(out);
        out = self.conv(out, maps, L_idx)
        out = torch.cat([out,x], 2)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, pooling_layer):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_planes)
        self.conv = nn.Linear(in_planes, out_planes);
        self.pooling = pooling_layer;
        self.dropout = nn.Dropout(0.2);

    def forward(self, x, id_list):
        out = x;
        out = out.permute(0,2,1).contiguous();
        out = self.bn(out);
        out = out.permute(0,2,1).contiguous();
        out = F.relu(out);
        out = self.dropout(out);
        out = self.conv(out)
        out = self.pooling(out, id_list);
        return out

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        if (args.sparse):
            self.index_list = data.id_list;
            
        self.growth_rate = growth_rate
        
        pooling_layers = [];
        for i in range(pool_num):
            pooling_layers += [Pooling(data.pooling_ks[i])];
        self.pooling_layers = nn.ModuleList(pooling_layers);
        
        x = 2 * growth_rate;
        self.conv1 = GraphConv(nodes_num_list[0], 3, x, 5, graph_num)
        
        self.layers1 = [];
        for i in range(blocks[0]):
            self.layers1.append(Bottleneck(x, self.growth_rate, 0))
            x += growth_rate;
        self.layers1 = nn.ModuleList(self.layers1);
        out_planes = x / 2;  
        self.trans1 = Transition(x, out_planes, self.pooling_layers[0]);
        x = out_planes;
        
        self.layers2 = [];
        for i in range(blocks[1]):
            self.layers2.append(Bottleneck(x, self.growth_rate, 1))
            x += growth_rate;
        self.layers2 = nn.ModuleList(self.layers2);
        out_planes = x / 2;  
        self.trans2 = Transition(x, out_planes, self.pooling_layers[1]);
        x = out_planes;
        
        self.layers3 = [];
        for i in range(blocks[2]):
            self.layers3.append(Bottleneck(x, self.growth_rate, 2))
            x += growth_rate;
        self.layers3 = nn.ModuleList(self.layers3);
        '''
        out_planes = x / 2;  
        self.trans3 = Transition(x, out_planes, self.pooling_layers[2]);
        x = out_planes;
        
        self.layers4 = [];
        for i in range(blocks[3]):
            self.layers4.append(Bottleneck(x, self.growth_rate, 3))
            x += growth_rate;
        self.layers4 = nn.ModuleList(self.layers4);
        '''
        self.linear = nn.Linear(x, 10);
    
    def dense(self, block_id, layers, x, maps, L_idx):
        for i in range(blocks[block_id]):
            x = layers[i](x, maps, L_idx);
        return x;
    
    def forward(self, inputs):
        x = inputs[0]; maps_list = inputs[1]; L_list = inputs[2]; 
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        
        x = self.conv1(x, maps_list[0], L_list[0]);
        x = self.dense(0, self.layers1, x, maps_list[0], L_list[0]);
        x = self.trans1(x, self.index_list[0]);
        x = self.dense(1, self.layers2, x, maps_list[1], L_list[1]);
        x = self.trans2(x, self.index_list[1]);
        x = self.dense(2, self.layers3, x, maps_list[2], L_list[2]);
        #x = self.trans3(x, self.index_list[2]);
        #x = self.dense(3, self.layers4, x, maps_list[3], L_list[3]);
        x = self.pooling_layers[2](x, self.index_list[2]);
        x = self.pooling_layers[3](x, self.index_list[3]);
        #x = self.pooling_layers[4](x, self.index_list[4]);
        x = x.view(batch_size, -1);
        x = self.linear(x);
        return x
    
        
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window 
        if (args.mask):
            self.w = self.w * 2;
        self.num_layers = args.num_layers;
        self.hid = args.hidCNN
        self.linears = [nn.Linear(self.w, self.hid) for i in range(self.num_layers+1)];
        self.linears = nn.ModuleList(self.linears);
        self.linears2 = [nn.Linear(self.hid, 1)  for i in range(self.num_layers+1)];
        self.linears2 = nn.ModuleList(self.linears2);
        self.output = None;
        
        self.dropout = nn.Dropout(args.dropout);
        
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
        
    def forward(self, inputs):
        x = inputs[0];L = inputs[1];
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        
        t = x.view(batch_size * self.m, self.w);
        s = self.linears2[0](F.relu(self.linears[0](t)));
        
        P = L
            
        for i in range(1, self.num_layers + 1):
            PP = P.expand(batch_size, *P.size());
            t = torch.bmm(PP, x);
            t = t.view(batch_size * self.m, self.w);
            s = s + self.linears2[i](F.relu(self.linears[i](t)))
            P = torch.mm(L, P);
        s = s.view(batch_size, self.m);
        if (self.output != None):
            s = self.output(s);
        return s
    
        
        
        
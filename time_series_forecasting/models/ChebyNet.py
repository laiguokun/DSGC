import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
            
        if args.num_layers == 1:
            self.linears = [nn.Linear(self.w, 1)]; 
            self.linears2 = [nn.Linear(self.w, 1)]
        else:
            self.linears = [nn.Linear(self.w, self.hid)];
            self.linears2 = [nn.Linear(self.w, self.hid)]
            self.linears3 = [nn.Linear(self.w, self.hid)]
            self.linears4 = [nn.Linear(self.w, self.hid)]
            for i in range(1, self.num_layers - 1):
                self.linears.append(nn.Linear(self.hid, self.hid))
                self.linears2.append(nn.Linear(self.hid, self.hid))
                self.linears3.append(nn.Linear(self.hid, self.hid))
                self.linears4.append(nn.Linear(self.hid, self.hid))
            self.linears.append(nn.Linear(self.hid, 1));
            self.linears2.append(nn.Linear(self.hid, 1));
            self.linears3.append(nn.Linear(self.hid, 1));
            self.linears4.append(nn.Linear(self.hid, 1));
        self.linears = nn.ModuleList(self.linears);
        self.linears2 = nn.ModuleList(self.linears2);
        self.linears3 = nn.ModuleList(self.linears3);
        self.linears4 = nn.ModuleList(self.linears4);
        self.output = None;
        
        self.dropout = nn.Dropout(args.dropout);
        
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
    
    def linear_layer(self, i, x, L, L2, L3, batch_size):
        in_feature = x.size(2);
        Lx = torch.bmm(L,x);
        Lx = Lx.view(batch_size * self.m, in_feature);
        Lx = self.linears[i](Lx);
        L2x = torch.bmm(L2,x);
        L2x = L2x.view(batch_size * self.m, in_feature);
        L2x = self.linears3[i](L2x);
        L3x = torch.bmm(L3,x);
        L3x = L3x.view(batch_size * self.m, in_feature);
        L3x = self.linears4[i](L3x);        
        x = x.view(batch_size * self.m, in_feature);
        x = self.linears2[i](x);
        x = x + Lx + L3x + L2x;
        x = x.view(batch_size, self.m, -1);
        x = self.dropout(x);
        return x;
        
    def forward(self, inputs):
        x = inputs[0];L = inputs[1];
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        L2 = 2 * torch.mm(L,L) - Variable(torch.eye(self.m).cuda());
        L3 = 2 * torch.mm(L,L2) - L;
        L = L.expand(batch_size, *L.size());
        L2 = L2.expand(batch_size, *L2.size());
        L3 = L3.expand(batch_size, *L3.size());
        for i in range(self.num_layers):
            x = self.linear_layer(i, x, L, L2, L3, batch_size);
            if (i != self.num_layers - 1):
                x = F.relu(x);
            
        if (self.output != None):
            x = self.output(x);
        return x
    
        
        
        
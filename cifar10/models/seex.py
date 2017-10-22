import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SEEX(nn.Module):
    def __init__(self, n, m):
        super(SEEX, self).__init__()
        self.r = 16;
        self.m = m;
        self.n = n;
        self.hid = self.m/self.r;
        self.linear1 = nn.Linear(self.m, self.hid);
        self.linear2 = nn.Linear(self.hid, self.m);
        self.pool = nn.AvgPool1d(self.n);
        self.bn = nn.BatchNorm1d(self.m)

    def forward(self, x):
        batch_size = x.size(0);
        x = x.permute(0,2,1).contiguous();
        z = self.pool(x).squeeze();
        z = F.relu(self.linear1(z));
        z = F.sigmoid(self.linear2(z));
        z = z.view(batch_size, self.m, 1);
        z = z.expand(batch_size, self.m , self.n);
        x = x * z;
        x = self.bn(x);
        x = x.permute(0,2,1).contiguous();
        
        return x
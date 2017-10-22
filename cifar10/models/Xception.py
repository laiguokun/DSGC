'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 1024, 1024, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(args.dropout);
        self.features = self._make_layers(cfg['VGG13'])
        self.classifier2 = nn.Linear(1024, 10)
        self.ch = data.ch;
        self.dim = data.dim

    def forward(self, inputs):
        x = inputs[0]; mask = inputs[1]
        batch_size = x.size(0);
        x = x.view(batch_size, self.ch, self.dim, self.dim);
        mask = mask.view(1, self.dim, self.dim);
        mask = mask.expand(batch_size, *mask.size());
        x = torch.cat((x,mask), 1);
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier2(out);
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 4
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [self.dropout];
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=1),
                           nn.Conv2d(x, x, kernel_size=3, padding=1, groups=x),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

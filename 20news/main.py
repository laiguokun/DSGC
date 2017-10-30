import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from Optim import Optim
import argparse
import random
import numpy as np;
import Loader
import timeit
from models import ChebyNet, DCNN, MoNet, GCN, DSGC;
from utils import Data_utility;
    
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, required=True,help='location of the data file')
parser.add_argument('--epochs', type=int, default=100,help='upper epoch limit')
parser.add_argument('--model', type=str, required=True,help='')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--save', type=str,  default='save/model.pt',help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True, help='use gpu or not')
parser.add_argument('--loc', type=str, default=None, help='load the location file')
parser.add_argument('--nn_num', type=int, default=32, help='number of the nearest neighbors')
parser.add_argument('--adjacency', type=str, default=None, help='use which kind of adjacency matrix as the input, None means do not use it')
parser.add_argument('--sparse', type=str, default=None, help='locations after sparsification')
parser.add_argument('--embed', type=int, default=0, help='use node2vec in graph convolution')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args = parser.parse_args()

max_iter = args.epochs;
batch_size = args.batch_size;

torch.cuda.set_device(args.gpu)
print('args', args)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

Data = Data_utility(args);

# setup model and optimizer
model = eval(args.model).Model(args, Data);
model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = Optim(model.parameters(), 'sgd', lr = 0.1, weight_decay=1e-4)
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

total_time = 0.0
best_loss = 0.0
state = None
for iteration in range(max_iter):
    # training
    model.train()
    train_res = 0.0
    time_st = timeit.default_timer()
    cnt = 0;
    for data in Data.get_batches([Data.x_train, Data.y_train], batch_size, shuffle=True):
        inputs = data[0]
        cnt += 1;
        if (len(data[1].size()) > 1 and data[1].size(1) == 1):
            targets = data[1].view(-1,);
        else:
            targets = data[1];

        optimizer.zero_grad()
        predict = model(inputs)
        output = loss(predict, targets)
        
        pred = predict.data.max(1)[1];
        correct = pred.eq(targets.data).cpu().sum()
        
        train_res += correct

        output.backward()
        optimizer.step()
    # evaluate by the validation set
    model.eval()
    valid_res = 0.0
    for data in Data.get_batches([Data.x_test, Data.y_test], batch_size):
        inputs = data[0]
        if (len(data[1].size()) > 1 and data[1].size(1) == 1):
            targets = data[1].view(-1,);
        else:
            targets = data[1];
        predict = model(inputs)
        pred = predict.data.max(1)[1];
        correct = pred.eq(targets.data).cpu().sum()
        
        valid_res += correct
    delta_time = (timeit.default_timer() - time_st)
    total_time += delta_time
    best_loss = valid_res
    
    print("[%3d/%4d] %.2f(m) train_loss %.6f valid_loss %.6f" %
          (iteration, max_iter, total_time / 60.0,
           train_res / (Data.train_num),
           valid_res / (Data.test_num)))
    if (iteration == 200 or iteration == 300 or iteration == 350):
        optimizer.updateLearningRate();
print('test: ', best_loss / (Data.test_num));

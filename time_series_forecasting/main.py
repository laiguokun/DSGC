import argparse
import math
import time

import torch
import torch.nn as nn
from models import LSTNet, AR, VAR, GCN, ChebyNet, MoNet, DSGC
import numpy as np;
import importlib
from utils import *;
import Optim

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        if (args.mask):
            mask = inputs[2];
        output = model(X);
        scale = loader.scale.expand(output.size(0), loader.m)
        if args.mask:
            n_samples += torch.sum(mask).data[0];
            total_loss += evaluateL2(output * scale * mask, Y * scale * mask).data[0]
            total_loss_l1 += evaluateL1(output * scale * mask, Y * scale * mask).data[0]
        else:
            total_loss += evaluateL2(output * scale , Y * scale ).data[0]
            total_loss_l1 += evaluateL1(output * scale , Y * scale ).data[0]
            n_samples += (output.size(0) * loader.m);

    rse = math.sqrt(total_loss / n_samples)/loader.rse
    rae = (total_loss_l1/n_samples)/loader.rae
    correlation = 0;
    return rse, rae, correlation;

def train(loader, data, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    t = 0;
    for inputs in loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        if (args.mask):
            mask = inputs[2];
        model.zero_grad();
        output = model(X);
        scale = loader.scale.expand(output.size(0), loader.m)
        if (args.mask):
            loss = criterion(output * scale * mask, Y * scale * mask);
        else:
            loss = criterion(output * scale, Y * scale);
        loss.backward();
        total_loss += loss.data[0];
        grad_norm = optim.step();
        if args.mask:
            n_samples += torch.sum(mask).data[0];
        else:
            n_samples += (output.size(0) * loader.m);
            
    return total_loss / n_samples

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, required=True,help='location of the data file')
parser.add_argument('--train', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--valid', type=float, default=0.2,help='how much data used for validation')
parser.add_argument('--model', type=str, default='LSTNet',help='')
# paramters for the LSTNet
parser.add_argument('--hidCNN', type=int, default=50, help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=50, help='number of RNN hidden units')
parser.add_argument('--num_layers', type=int, default=10, help='number of hidden layers')
parser.add_argument('--CNN_kernel', type=int, default=6,help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,help='The window size of the highway component')
parser.add_argument('--skip', type=int, default=24, help='skip-length in the LSTNet')
parser.add_argument('--hidSkip', type=int, default=5, help='hidden units nubmer of skip layer')

parser.add_argument('--L1Loss', action='store_true', default=False, help='the loss function used')
parser.add_argument('--window', type=int, default=24 * 7,help='window size')
parser.add_argument('--clip', type=float, default=1.,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--save', type=str,  default='save/model.pt',help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True, help='use gpu or not')
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 regularization)')
parser.add_argument('--horizon', type=int, default=1, help='predict horizon')
parser.add_argument('--normalize', type=int, default=0, help='the normalized method used, detail in the utils.py')
parser.add_argument('--output_fun', type=str, default=None, help='the output function of neural net')
parser.add_argument('--loc', type=str, default=None, help='load the location file')
parser.add_argument('--mask', action='store_true', default=False, help='record the missing points, designed for USHCN dataset')
parser.add_argument('--metric', type=int, default=0, help='use rse metric or rmse')
parser.add_argument('--nn_num', type=int, default=0, help='number of the nearest neighbors')
parser.add_argument('--adjacency', type=str, default=None, help='use which kind of adjacency matrix as the input, None means do not use it')
parser.add_argument('--embed', type=int, default=0, help='use node2vec in graph convolution')
parser.add_argument('--sigma', action='store_true', default=False, help='learn the sigma of the graph by the network')

args = parser.parse_args()
print(args);
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args);
print('buliding model')
model = eval(args.model).Model(args, Data);

if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();


best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, weight_decay = args.weight_decay,
)
test_acc, test_rae, test_corr = 0, 0, 0
# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training');
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train, model, criterion, optim, args.batch_size)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.8f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        if val_loss < best_val:
            #with open(args.save, 'wb') as f:
            #    torch.save(model.state_dict(), f)
            best_val = val_loss
            print('best validation');
            test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
            print ("test rse {:6.5f} | test rae {:6.5f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
print(test_acc, test_rae, test_corr);

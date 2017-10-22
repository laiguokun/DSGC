import torch
import numpy as np;
from torch.autograd import Variable
from graph import *;


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args):
        self.cuda = args.cuda;
        self.P = args.window;
        self.h = args.horizon
        #use which kind of the adjacency matrix as input ( Laplaican, or normal or None)
        self.adjacency = args.adjacency; 
        fin = open(args.data);
        self.rawdat = np.loadtxt(fin,delimiter=',');
        self.embed = args.embed
        print('data shape', self.rawdat.shape);
        # if data has missing points
        self.create_mask(args);
        # load location file
        self.load_location(args);

        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1));
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m = self.dat.shape;
        self.normalize = args.normalize
        self.scale = np.ones(self.m);
        self._normalized(self.normalize);

        if (args.nn_num > 0):
            self.compute_nn_list(args.nn_num)

        self._split(int(args.train * self.n), int((args.train+args.valid) * self.n), self.n);

        self.scale = torch.from_numpy(self.scale).float();

        #compute denominator of the RSE and RAE
        self.compute_metric(args);
        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

    def compute_nn_list(self, nn_num):
        print('begin building graph');
        self.nn_num = nn_num;
        self.dist, self.nn_list = distance_sklearn_metrics(self.locations, k = nn_num);
        if (self.adjacency == 'Laplacian'):
            '''
            self.adj = adjacency(self.dist, self.nn_list);
            self.adj = laplacian(self.adj);
            self.adj = torch.Tensor(self.adj.toarray());
            '''
            self.adj = adjacency_gcn(self.dist, self.nn_list);
            if (self.cuda):
                self.adj = self.adj.cuda();
                
        if (self.adjacency == 'normal4'):
            self.adjxp, self.adjxn, self.adjyp, self.adjyn = adjacency_xy4(self.dist, self.nn_list, self.locations);
            if (self.cuda):
                self.adjxp = self.adjxp.cuda();
                self.adjxn = self.adjxn.cuda();
                self.adjyp = self.adjyp.cuda();
                self.adjyn = self.adjyn.cuda();
        
        if (self.adjacency == 'normal'):
            self.adjx, self.adjy = adjacency_xy(self.dist, self.nn_list, self.locations);
            if (self.cuda):
                self.adjx = self.adjx.cuda();
                self.adjy = self.adjy.cuda();
        
        if (self.adjacency == 'randomwalk'):
            self.adj = adjacency_randomwalk(self.dist, self.nn_list, self.locations)
            if (self.cuda):
                self.adj = self.adj.cuda();
        
        if (self.adjacency == 'sigma'):
            self.sigma, self.adj = adjacency_sigma(self.dist, self.nn_list, self.locations)
            self.sigma = 1.
            if (self.cuda):
                #self.sigma = self.sigma.cuda();
                self.adj = self.adj.cuda();
        
        if (self.adjacency == '2d'):
            self.L_idx, self.adj = adjacency_2d(self.dist, self.nn_list, self.locations)
            if (self.cuda):
                self.L_idx = self.L_idx.cuda();
                self.adj = self.adj.cuda();
                
        self.nn_list = torch.from_numpy(self.nn_list).long();
        self.idx = torch.arange(0, self.m).long()
        if (self.cuda):
            self.nn_list = self.nn_list.cuda();
            self.idx = self.idx.cuda();
        self.nn_list = Variable(self.nn_list);
        self.idx = Variable(self.idx);
        print('finish building graph');

    def load_location(self, args):
        self.loc = args.loc
        if (args.loc == None):
            return;
        fin = open(args.loc);
        self.locations = np.loadtxt(fin, delimiter=',');

    def create_mask(self, args):
        self.mask = args.mask;
        if (args.mask == False):
            return;
        self.mm = (self.rawdat != -9999);
        self.rawdat[(self.rawdat == -9999)] = 0.;
        self.rawdat[(self.rawdat == 9999)] = 0.;
        self.mm = self.mm.astype('uint8');
        size = self.rawdat.shape[0] * self.rawdat.shape[1];
        print('missing ratio {}'.format(1 - float(np.sum(self.mm))/size));

    def compute_metric(self, args):
        #use the normal rmse and mae when args.metric == 0;
        if (args.metric == 0):
            self.rse = 1.;
            self.rae = 1.;
            return;

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m);
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));



    def _normalized(self, normalize):
        #normalized by the maximum value of entire matrix.
        if (normalize == 0):
            self.dat = self.rawdat
            
        print(np.mean(np.abs(self.rawdat)) * 2, np.max(np.abs(self.rawdat)))
        if (normalize == 1):
            self.scale = self.scale * (np.mean(np.abs(self.rawdat))) * 2
            self.dat = self.rawdat / (np.mean(np.abs(self.rawdat)) * 2);

        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]));
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]));
        
        self.mean = np.mean(np.abs(self.dat))

    def _split(self, train, valid, test):

        train_set = range(self.P+self.h-1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.n);
        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);
        self.test = self._batchify(test_set, self.h);
        if (train==valid):
            self.valid = self.test


    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        if (self.mask):
            X = torch.zeros((n,self.P*2, self.m));
        else:
            X = torch.zeros((n, self.P, self.m));
        Y = torch.zeros((n, self.m));
        mask = torch.zeros((n, self.m));
        
        for i in range(n):
            end = idx_set[i] - self.h + 1;
            start = end - self.P;
            X[i,:self.P,:] = torch.from_numpy(self.dat[start:end, :]);
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :]);
            if (self.mask):
                X[i,-self.P:,:] = torch.from_numpy(self.mm[start:end, :]);
                mask[i,:] = torch.from_numpy(self.mm[idx_set[i],:]).long();

        return [X, Y, mask];

    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]; targets = data[1];
        masks = data[2];
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; Y = targets[excerpt];
            mask = masks[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
                mask = mask.cuda();
            
            model_inputs = Variable(X);
            if (self.adjacency == 'Laplacian' or self.adjacency == 'randomwalk'):
                model_inputs = [model_inputs] + [Variable(self.adj)];
            if (self.adjacency == '2d'):
                model_inputs = [model_inputs] + [Variable(self.adj), self.L_idx];
            if (self.embed):
                model_inputs = model_inputs + [self.idx]
            data = [model_inputs, Variable(Y), Variable(mask)]
            yield data;
            start_idx += batch_size

import torch
import numpy as np;
from torch.autograd import Variable
from graph import *;
import math;


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args):
        self.cuda = args.cuda;
        self.adjacency = args.adjacency; 
        f = np.load(args.data)
        self.x_train, self.y_train = f['x_train'], f['y_train']
        self.x_test, self.y_test = f['x_test'], f['y_test']
        
        self.x_train = torch.from_numpy(np.asarray(self.x_train)).float()/255.0;
        self.y_train = torch.from_numpy(np.asarray(self.y_train)).long();
        self.x_test = torch.from_numpy(np.asarray(self.x_test)).float()/255.0;
        self.y_test = torch.from_numpy(np.asarray(self.y_test)).long();
        print(self.y_train.size());
        self.train_num = self.y_train.size(0);
        self.test_num = self.y_test.size(0);
        
        self.load_location(args);

        self.n, self.w, self.m = self.x_train.size();
        self.ch = self.w;
        if (self.ch == 3):
            self.dim = 32;
        self.mask = args.mask
        if (self.mask):
            self.create_mask();
            
        self.sparse = args.sparse;
        print(self.n, self.w, self.m);

        if (args.nn_num > 0 and args.sparse == None):
            self.compute_nn_list(args.nn_num)
        
        if (args.sparse != None):
            self.compute_sparse_nn_list(args);
            
    def create_mask(self):
        self.mm = torch.zeros(self.dim, self.dim);
        self.tmp_train = torch.zeros(self.train_num, self.ch, self.dim, self.dim);
        self.tmp_test = torch.zeros(self.test_num, self.ch, self.dim, self.dim);
        for i in range(self.m):
            x, y = self.locations[i][0], self.locations[i][1]
            self.mm[x, y] = 1;
            self.tmp_train[:, :, x, y] = self.x_train[:,:,i];
            self.tmp_test[:,:,x,y] = self.x_test[:,:,i];
        self.x_train = self.tmp_train;
        self.x_test = self.tmp_test;
        if (self.cuda):
            self.mm = self.mm.cuda();
            
    def compute_nn_list(self, nn_num):
        print('begin building graph');
        self.nn_num = nn_num;
        self.dist, self.nn_list = distance_sklearn_metrics(self.locations, k = nn_num);
        if (self.adjacency == 'Laplacian'):
            self.adj = adjacency_gcn(self.dist, self.nn_list);
            if (self.cuda):
                self.adj = self.adj.cuda();
        
        if (self.adjacency == 'randomwalk'):
            self.adj = adjacency_randomwalk(self.dist, self.nn_list, self.locations)
            if (self.cuda):
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
    
    def compute_sparse_nn_list(self, args):
        fin = np.load(args.sparse);
        self.locations_list = fin['locations']
        self.L_list = [];
        self.adj_list = []
        self.id_list = fin['index'];
        self.pooling_ks = []
        for i in range(len(self.id_list)):
            self.id_list[i] = Variable(torch.from_numpy(self.id_list[i]).long().cuda())
            self.pooling_ks += [self.id_list[i].size(1)]
            
        if self.m == 256:
            #k-nearest setting for the subsampled dataset
            self.nn_cfgs = [16, 12, 8, 3];
        else:
            #k-nearest setting for the origin dataset
            self.nn_cfgs = [8, 8, 8, 8, 3];
            
        for i in range(len(self.nn_cfgs)):
            locations = self.locations_list[i];
            dist, nn_list = distance_sklearn_metrics(locations, k = self.nn_cfgs[i]);
            if (self.adjacency == '2d'):
                L_idx, adj = adjacency_2d(dist, nn_list, locations)
                self.L_list.append(L_idx.cuda());
                self.adj_list.append(Variable(adj.cuda()));
            if (self.adjacency == 'Laplacian'):
                adj = adjacency_gcn(dist, nn_list);
                self.L_list.append(Variable(adj.cuda()));
            

    def load_location(self, args):
        self.loc = args.loc
        if (args.loc == None):
            return;
        fin = np.load(args.loc);
        self.locations = fin['loc']
        self.locations = self.locations.astype(np.float);

    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]; targets = data[1];
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
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            model_inputs = Variable(X);
            if (self.mask):
                model_inputs = [model_inputs] + [Variable(self.mm)];
            if (self.sparse == None):
                if (self.adjacency == 'Laplacian' or self.adjacency == 'randomwalk'):
                    model_inputs = [model_inputs] + [Variable(self.adj)];
                if (self.adjacency == '2d'):
                    model_inputs = [model_inputs] + [Variable(self.adj), self.L_idx];
            else:
                if (self.adjacency == '2d' and self.sparse != None):
                    model_inputs = [model_inputs] + [self.adj_list, self.L_list, self.id_list]
                if (self.adjacency == 'Laplacian' or self.adjacency == 'randomwalk'):
                    model_inputs = [model_inputs] + [self.L_list];
            data = [model_inputs, Variable(Y)]
            yield data;
            start_idx += batch_size

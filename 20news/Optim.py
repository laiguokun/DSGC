import math
import torch.optim as optim

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay = self.weight_decay, momentum=0.9)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr=0.1, weight_decay = 0.):
        self.params = list(params)  
        self.lr = lr
        self.weight_decay = weight_decay;
        self.method = method
        self._makeOptimizer()

    def zero_grad(self):
        self.optimizer.zero_grad();
        
    def step(self):
        self.optimizer.step()

    def updateLearningRate(self):
        self.lr = self.lr * 0.1
        self._makeOptimizer()

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
import os
import logging
from .util import metric

class CommonLayer(nn.Module, ABC):
    #Architecture of a common linear (it has to be inherited using say linear layer)
    def __init__(self, ip_size, op_size, act): #Parameters - act : activation function; ip_size : input size  op_size : output size
        super(CommonLayer, self).__init__()
        self._ip_size = ip_size
        self._op_size = op_size
        self.act = act
        self.model = None # this will be defined later in the init_layer function

    def init_layer(self, **kwargs):
        # Initializes the model by setting the input and output sizes and setup the layer that transforms the input to the output
        if self._ip_size < 0:
            self._ip_size = kwargs['ip_size']
        if self._op_size < 0:
            self._op_size = kwargs['op_size']
        assert self._ip_size > 0 and self._op_size > 0, "sizes are not valid"
        self.model = self.layer()(self._ip_size, self._op_size) # create a model with a layer (could be linear) that takes input and return output

    def forward(self, x): # Forwards x through the model
        y = self.model(x)
        return self.act(y)

    def get_shape(self): # Returns the shape of the model
        return (self._ip_size, self._op_size)

    def weights_init(self): # Initializes the weights of the model
        nn.init.xavier_uniform_(self.model.weight) # xavier_uniform_, layr.weight.data.fill_(0.01)

    @abstractmethod
    def layer(self): # Abstract method for a layer (could be simply a linear layer nn.Linear)
        pass

class LinearLayer(CommonLayer):
    # A LinearLayer 
    def __init__(self, ip_size, op_size, act=None):
        if act is None:
            act = nn.ReLU()
        super(LinearLayer, self).__init__(ip_size, op_size, act)
        self.init_layer()

    def layer(self):
        return nn.Linear

class BaseNNModule(nn.Module):
    def __init__(self, lst_structure):
        super(BaseNNModule, self).__init__()
        self.layrs = nn.ModuleList()
        for lyr in lst_structure:
            self.layrs.append(LinearLayer(lyr['input_size'], lyr['output_size'], act=eval(lyr['act'])))

    def weights_init(self):
        # Initialize the weights of the model
        for lyr in self.layrs:
            lyr.weights_init()
            # nn.init.xavier_uniform_(mdl.weight)

    def fit(self, data_iterator, optimizer, criterion): # Optimizes the model using the input data. It passes over the data only once
        # Parameters: data_iterator: iterator that returns batches of x and y;  optimizer: pytorch optimizer to be used to optimize the network criterion: loss function

        self.train() # set model to training mode
        # compute sum of losses over all batches
        running_loss = 0
        while True:
            try:
                x_batch, y_batch, base_batch = next(data_iterator)
                # compute ypred (forward pass)
                y_pred = self.forward(x_batch, base_batch)
                loss = criterion(y_pred, y_batch)
                optimizer.zero_grad() # init gradients to zeros
                loss.backward() # backpropagation (backward pass)
                optimizer.step() # update the weights
                running_loss += loss.data # detach().numpy()
            except StopIteration:
                # if StopIteration is raised, break from loop
                break
        logging.debug(" running loss is %.4f" % (running_loss))
        return y_pred, running_loss

    def predict(self, data_loader):  # Predicts the test data
        # Parameters: x_data: input data
        self.eval() # set model to evaluation mode
        x_data, _, base_pred = data_loader.full_data_iterator()
        y_pred = self.forward(x_data, base_pred)
        return y_pred

    def score(self, data_loader): # Returns the prediction score (accuracy) of the model on the data_iterator
        # Parameters: data_iterator: iterator that returns batches of x and y ; metrics: dictionary of functions to compute required metric
        self.eval() # set model to evaluation mode
        summary = []
        _, y, _ = data_loader.full_data_iterator()
        y_pred = self.predict(data_loader)
        metrics_mean = metric(y_pred.detach().numpy(), y.detach().numpy())
        logging.debug("- Eval metrics : " + str(metrics_mean))
        return metrics_mean

class GateEnsemble(BaseNNModule):
    def __init__(self, lst_structure):
        super(GateEnsemble, self).__init__(lst_structure)

    def forward(self, x, base): # x: is the original input data; base: is the prediction from the individual models
        w = x
        for lyr in self.layrs:
            w = lyr(w) # the last layer should be softmax
        y = w*base # this is element-wise multiplication
        y = torch.sum(y, dim=1) # sum over all columns (i.e. for each row)
        return y.view(-1,)

class FFN(BaseNNModule):
    def __init__(self, lst_structure):
        super(FFN, self).__init__(lst_structure)

    def forward(self, x, base): # x: is the original input data, # base: is the prediction from the individual models
        y = x
        for lyr in self.layrs:
            y = lyr(y) # the last layer should be softmax
        return y.view(-1,)
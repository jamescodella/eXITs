from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
import torch.autograd as py

class DataLoader():
    def __init__(self, data_file):
        super(DataLoader, self).__init__()
        self.data_file = data_file
        self.load_data()

    def load_data(self):
        boston = load_boston()
        y = boston['target'].copy()
        ypred1 = boston['target'].copy() # prediciton from model 1
        idx = random.sample(range(y.shape[0]), int(.2 * y.shape[0]))
        noise = np.random.normal(0,np.std(y),len(idx)).astype(int)
        ypred1[idx] = ypred1[idx] + noise
        ypred2 = boston['target'].copy() # prediciton from model 2
        idx = random.sample(range(y.shape[0]), int(.2 * y.shape[0]))
        noise = np.random.normal(0,np.std(y),len(idx)).astype(int)
        ypred2[idx] = ypred2[idx] + noise
        ypred3 = boston['target'].copy() # prediciton from model 3
        idx = random.sample(range(y.shape[0]), int(.2 * y.shape[0]))
        noise = np.random.normal(0,np.std(y),len(idx)).astype(int)
        ypred3[idx] = ypred3[idx] + noise
        X = boston['data']
        N = X.shape[0]
        X_train, self.X_test, y_train, self.y_test, y_pred1_train, self.y_pred1_test, y_pred2_train, self.y_pred2_test, y_pred3_train, self.y_pred3_test = train_test_split(X, y, ypred1, ypred2, ypred3, test_size=0.20, random_state=100)
        self.X_train, self.X_val, self.y_train, self.y_val, self.y_pred1_train, self.y_pred1_val, self.y_pred2_train, self.y_pred2_val, self.y_pred3_train, self.y_pred3_val = train_test_split(X_train, y_train, y_pred1_train, y_pred2_train, y_pred3_train, test_size=0.20, random_state=100)

    def get_training(self):
        return self.X_train, self.y_train

    def get_ntraining(self): # return number of data points in the training set

        return self.X_train.shape[0]

    def get_test(self):
        return self.X_test, self.y_test

    def get_ntest(self): # return number of data points in the test set
        return self.X_test.shape[0]

    def get_val(self):
        return self.X_val, self.y_val

    def get_nval(self):         # return number of data points in the validation set  
        return self.X_val.shape[0]

    def get_pred_training(self): # return the prediction from the 3 base models in the training set  

        base_pred = np.reshape(np.concatenate((self.y_pred1_train, self.y_pred2_train, self.y_pred3_train)), (-1,3), order='F') # order='F' to go through rows first and then columns
        return base_pred

    def get_pred_test(self):
        # return the prediction from the 3 base models in the test set 
        base_pred = np.reshape(np.concatenate((self.y_pred1_test, self.y_pred2_test, self.y_pred3_test)), (-1,3), order='F')
        return base_pred

    def get_pred_val(self):
        # return the prediction from the 3 base models in the validation set 
        base_pred = np.reshape(np.concatenate((self.y_pred1_val, self.y_pred2_val, self.y_pred3_val)), (-1,3), order='F')
        return base_pred

    def get_shape(self):
        return (self.X_train.shape[0], self.X_train.shape[1])

class StaticDataLoader(): #Loads and iterates over the non-temporal data    
    def __init__(self, x_data, y_data, x_pred, params): # batch_size, cuda
        super(StaticDataLoader, self).__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.base_pred = x_pred # prediction from the base models
        self.params = params

    def get_number_examples(self):
        return self.x_data.shape[0]

    def data_iterator(self, shuffle=True): # Iterates over batches of x_data. The function set_data_set has to be called to set which data to operate on
        # Parameters: shuffle: shuffles the data before batchfying it 
        sz = self.x_data.shape[0]
        order = list(range(sz))
        if shuffle:
            random.seed(230)
            random.shuffle(order)
        # one pass over data
        batch_size = self.params['batch_size']
        for i in range(sz // batch_size + 1):
            if i == (sz // batch_size): # last batch
                batch_block = order[i*batch_size : ]
                batch_size = sz - batch_size * (sz // batch_size)
            else :
                batch_block = order[i*batch_size : (i+1)*batch_size]
            x_batch = np.zeros(shape=(batch_size, self.x_data.shape[1]))
            base_batch = np.zeros(shape=(batch_size, self.base_pred.shape[1]))
            y_batch = np.zeros(shape=(batch_size, ))  # self.y_data.shape[1]
            for newidx, orgidx in enumerate(batch_block):
                x_batch[newidx,:] = self.x_data[orgidx, :]
                base_batch[newidx,:] = self.base_pred[orgidx, :]
                if self.y_data is not None:
                    y_batch[newidx,] = self.y_data[orgidx,]
            x_batch, y_batch, base_batch = torch.Tensor(x_batch), torch.FloatTensor(y_batch), torch.FloatTensor(base_batch) # .view(-1, T)
            if self.params['cuda']: x_batch, y_batch, base_batch = x_batch.cuda(), y_batch.cuda(), base_batch.cuda() # shift tensors to GPU if available
            # convert them to Variables to record operations in the computational graph
            x_batch, y_batch, base_batch = py.Variable(x_batch), py.Variable(y_batch), py.Variable(base_batch)
            yield x_batch, y_batch, base_batch

    def full_data_iterator(self):
        # It does not do batches but returns the entire data but in pytorch format
        y_batch = None
        x_batch, base_batch = torch.Tensor(self.x_data),  torch.FloatTensor(self.base_pred) # .view(-1, T)
        if self.y_data is not None: y_batch = torch.FloatTensor(self.y_data)
        if self.params['cuda']: # Shift tensors to GPU if available
            x_batch,  base_batch = x_batch.cuda(),  base_batch.cuda()
            if self.y_data is not None:
                y_batch = y_batch.cuda()
        # convert them to Variables to record operations in the computational graph
        x_batch, base_batch = py.Variable(x_batch),  py.Variable(base_batch)
        if self.y_data is not None:
            y_batch = py.Variable(y_batch)
        return x_batch, y_batch, base_batch

if __name__ == "__main__":
    data_loader = DataLoader("")
    print (data_loader.get_ntraining())
    print (data_loader.get_shape())
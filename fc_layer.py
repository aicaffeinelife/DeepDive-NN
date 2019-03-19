import numpy  as np 
from layer import BaseLayer
from activations import *

class FCLayer(BaseLayer):
    '''
    Instantiate a fully connected layer. 

    Args:
    :in_units:-> The dimensions in the input 
    :out_units:-> The dimensions in the output layer 
    :activation:->Sigmoid or Relu or any other you define in activations.py 
    :initilazer:-> The type of initialization to use. Currently random normal.
    '''
    def __init__(self, in_units, out_units, activation=None, initializer='Normal'):
        self._in_units = in_units 
        self._out_units = out_units 
        self.act = activation
        self._init = initializer
        self.setupLayer()


    def setupLayer(self):
        '''
        Initialize the weights and the biases 
        of the particular layer of the given shape. 
        If input: NxD then, 
        W: DxH 
        b: H 
        where H is the hidden dimension
        '''
        if self._init != 'Normal':
            raise ValueError('Currently only gaussian initializer is supported')

        self.W = np.random.randn(self._in_units, self._out_units)
        self.b = np.zeros([self._out_units])


    def forward_pass(self, x_input, cache=None):
        '''
        Compute the forward pass 
        '''
        y = np.matmul(x, self.W) + self.b 
        if self.act is not None:
            y_ = self.act().forward(y) 
        return y_



if __name__ == '__main__':
    fc = FCLayer(784, 1200, ReLU)
    x = np.random.randn(1, 784)
    y_ = fc.forward_pass(x)
    print('FC output')
    print('---'*4)
    print(y_)
    print(y_.shape)
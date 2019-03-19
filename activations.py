''' Definitions of activation functions ''' 
import numpy as np 


class BaseActivation(object):
    def __init__(self):
        self.cache = None # cache of variables to compute the backward pass

    def forward(self, x):
        raise NotImplementedError('Forward method needs to be defined')

    def backward(self, input, cache=None):
        raise NotImplementedError('Backward method needs to be defined')



class Sigmoid(BaseActivation):
    '''
    Sigmoid non linearity. 
    '''
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        y = 1./(1. + np.exp(-x))
        return y 

    def backward(self, dout, cache=None):
        '''
        Compute the backward pass through 
        sigmoid nonlinearity. If a cache 
        of variables is passed in, simply 
        store that in a cache.
        '''
        grad = self.forward(dout)*self.forward(1-dout)
        if cache is not None:
            self.cache = cache
        return grad 




class ReLU(BaseActivation):
    """ReLU nonlinearity.
    """
    def __init__(self):
        super(ReLU, self).__init__()
    

    def forward(self, x):
        return np.max(x,0)

    def backward(self, dout, cache=None):
        '''
        Backward pass for relu non linearity
        '''
        if cache is not None:
            self.cache = cache 
        dout[dout<=0] = 0 
        dx = dout 
        return dx 



        


if __name__ == '__main__':
    x = np.random.randn(10,10)
    sigmoid = Sigmoid()
    y = sigmoid.forward(x)
    
    print('Input:{}'.format(x))
    print('Forward:{}'.format(y))




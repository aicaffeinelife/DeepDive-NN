''' Abstract layer to be implemented by different types of layers ''' 

class BaseLayer(object):
    def forward_pass(self, x_input, cache=None):
        raise NotImplementedError("Forward pass must be declared by a layer")

    def backward_pass(self, dout, cache=None):
         raise NotImplementedError("Forward pass must be declared by a layer")

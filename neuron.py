import numpy as np 



class Neuron(object):
    def __init__(self, x):
        self._input = x


    def forward(self, thresh):
        if self._input >= thresh:
            return 1 
        else:
            return 0

    

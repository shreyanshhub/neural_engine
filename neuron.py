from scalar import *
import random
import numpy as np

class Neuron:

    def __init__(self,nin):
        self.w = [Scalar(random.uniform(-1,1)) for i in range(nin)]
        self.b = Scalar(random.uniform(-1,1))

    def __call__(self,x):
        activation = sum((wj*xj for wj,xj in zip(self.w,x)),self.b)
        output = activation.tanh()
        return output

    def parameters(self):

        return self.w + [self.b]
        

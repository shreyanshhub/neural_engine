from scalar import *
import random
import numpy as np

class Neuron:

  def __init__(self,nin):

    self.w = [Scalar(random.uniform(-1,1)) for i in range(nin)]
    self.b = random.uniform(-1,1)

  def __call__(self,x):

    out = sum(w_i*x_i for w_i,x_i in zip(self.w,x))
    activation = (out+self.b).tanh()
    return activation 

  def parameters(self):

    return self.w + [self.b]

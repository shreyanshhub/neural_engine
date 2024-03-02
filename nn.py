from layer import *

class NN:

  def __init__(self,nin,nouts):

    size = [nin] + nouts
    self.layers = [Layer(size[i],size[i+1]) for i in range(len(nouts))]

  def __call__(self,x):

    for layer in self.layers:
      x = layer(x)

    return x

  def parameters(self):

    return [parameter for layer in self.layers for parameter in layer.parameters()]

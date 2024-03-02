from neuron import *

class Layer:

  def __init__(self,nin,nouts):

    self.neurons = [Neuron(nin) for i in range(nouts)]

  def __call__(self,x):
    
    output = [unit(x) for unit in self.neurons]
    return output[0] if len(output) == 1 else output

  def parameters(self):

    return [parameter for unit in self.neurons for parameter in unit.parameters()]

  

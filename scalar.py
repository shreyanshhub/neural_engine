from vis import *

class Scalar:
    
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Scalar(data={self.data})"
    
    def __add__(self,other):
        
        
        other = other if isinstance(other,Scalar) else Scalar(other)
        out = Scalar(self.data+other.data,(self,other),'+')
        
        def _backward():
            
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        
        out._backward = _backward
            
        return out
    
    def __mul__(self,other):
        
        other = other if isinstance(other,Scalar) else Scalar(other)
        out = Scalar(self.data*other.data,(self,other),'*')
        
        def _backward():
            
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return self*other
    
    def exp(self):
        out = Scalar(math.exp(self.data),(self,),'exp')
        
        def _backward():
            self.backward += out.data*out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self,other):
        
        assert isinstance(other,(int,float)),'support float/int'
        out = Scalar(self.data**other,(self,),f'**{other}')
        
        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self,other):
        return self*other**-1
    
    def __neg__(self):
        return self*-1
    
    def __sub__(self,other):
        return self + (-other)
    
    def tanh(self):
        n = self.data
        ot = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        
        out = Scalar(ot,(self,),'tanh')
        
        def _backward():
            self.grad += (1 - ot**2)*out.grad
        out._backward = _backward
        return out
        
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
          node._backward()

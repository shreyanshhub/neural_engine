# neural_engine
neural network  built on a scalar engine rather than regular tensor engine

## scalar.py

This is a simple Python class named `Scalar` which represents scalar values and supports basic arithmetic operations and some elementary mathematical functions along with automatic differentiation for gradient computation.

### Features

- Initialization with a scalar value, children nodes, operation type, and label.
- Overloaded arithmetic operations: addition, multiplication, exponentiation, division, subtraction, and negation.
- Elementary mathematical functions: exponential (`exp`) and hyperbolic tangent (`tanh`).
- Automatic differentiation for gradient computation using backpropagation.
- Support for building computational graphs and computing gradients for scalar-valued expressions.

### Usage

You can use the `Scalar` class to perform arithmetic operations, compute mathematical functions, and automatically calculate gradients. Here's a brief overview of how to use it:

```python
# Importing the Scalar class
from scalar import Scalar
import math

# Creating scalar objects
a = Scalar(2)
b = Scalar(3)

# Arithmetic operations
c = a + b
d = a * b
e = a ** 2
f = a / b

# Mathematical functions
g = a.exp()
h = b.tanh()

# Implementing backprop
h.backward()

# Accessing values and gradients
print(c.data, c.grad)
print(d.data, d.grad)
print(e.data, e.grad)
print(f.data, f.grad)
print(g.data, g.grad)
print(h.data, h.grad)

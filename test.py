from ffnn import *
import matplotlib.pyplot as plt


f = NN(3,[4,3,1])

test_data = [[random.uniform(-1,1) for i in range(3)] for _ in range(10)]
labels = [random.choice([0,1]) for i in range(len(test_data))]

pred = [f(data) for data in test_data]

#implementing Stochastic Gradient Descent

epochs = 10
eta = 0.01

losses = []

for i in range(epochs):

    print("---------------Epoch -------------------",i)

    # Forward propagation
    pred = [f(data) for data in test_data]
    loss = sum((y_pred-y_true)**2 for y_pred,y_true in zip(pred,labels))
    losses.append(loss.data)

    print('Loss ',i,loss)

    for parameter in f.parameters():
        parameter.grad = 0.0

    # Backward propagation

    loss.backward()

    # Updating parameters

    for parameter in f.parameters():
        parameter.data += -eta * parameter.grad


plt.plot([i for i in range(epochs)],losses)
plt.xlabel('epoch')
plt.ylabel('loss')

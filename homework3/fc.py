import math
import numpy as np  
from download_mnist import load
import operator  
import time 
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,784)
x_test  = x_test.reshape(10000,784)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
np.random.seed(42)

def one_hot(Y):
    encoded_Y = np.zeros((Y.size, 10))
    encoded_Y[np.arange(Y.size), Y] =1
    return encoded_Y

def ReLu(Z):
    result = np.maximum(0,Z)
    return result

def softmax(Z):
    exp_z = np.exp(Z-np.max(Z, axis = 1, keepdims = True)) #this subtraction is done so it does not get calculated as "infinity"
    prob = exp_z/(np.sum(exp_z, axis = 1, keepdims = True))
    return prob

def foward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = ReLu(Z1)
    Z2 = np.dot(A1,W2) + b2
    A2 = ReLu(Z2)
    Z3 = np.dot(A2,W3) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def back_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, W2, W3):
    N = X.shape[0]

    dZ3 = A3 - Y
    dW3 = (1/N) * np.dot(A2.T, dZ3)
    db3 = (1/N) * np.sum(dZ3, axis = 0, keepdims = True)

    dZ2 = np.dot(dZ3, W3.T) * (Z2 > 0)
    dW2 = (1/N) * np.dot(A1.T, dZ2)
    db2 = (1/N) * np.sum(dZ2, axis = 0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
    dW1 = (1/N) * np.dot(X.T, dZ1)
    db1 = (1/N) * np.sum(dZ1, axis = 0, keepdims = True)

    return dW1 , db1, dW2, db2, dW3, db3

def cross_entropy_loss(A3, Y):
    N = Y.shape[0]
    loss = -np.sum(Y*np.log(A3+1e-8))/N
    return loss

def train(X_train, Y_train, y_train, W1, b1, W2, b2, W3, b3, alpha, epochs, batch_size):
   
    N = X_train.shape[0] #number of samples
    training_time_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()

        shuffling = np.random.permutation(N)
        X_shuffled = X_train[shuffling]
        Y_shuffled = Y_train[shuffling]

        for i in range(0, N, batch_size):
            #slicing the batches
            batch_X = X_shuffled[i: i + batch_size] 
            batch_Y = Y_shuffled[i: i + batch_size]

            Z1, A1, Z2, A2, Z3, A3 = foward_propagation(batch_X, W1, b1, W2, b2, W3, b3)
            dW1 , db1, dW2, db2, dW3, db3 = back_propagation(batch_X, batch_Y, Z1, A1, Z2, A2, Z3, A3, W2, W3)

            #update parameters
            W1 -= alpha*dW1
            W2 -= alpha*dW2
            W3 -= alpha*dW3

            b1 -= alpha * db1
            b2 -= alpha *db2
            b3 -= alpha * db3

        #prints the loss after each epoch 
        _, _, _, _, _, A3_train = foward_propagation(X_train, W1, b1, W2, b2, W3, b3)
        current_loss = cross_entropy_loss(A3_train, Y_train)

        predictions = np.argmax(A3_train, axis =1)
        accuracy = np.mean(predictions == y_train) *100

        epoch_duration = time.time()-epoch_start
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {current_loss:.4f} - Accuracy: {accuracy:.4f} - time: {epoch_duration:.2f}")
    
    training_time = time.time() - training_time_start
    print(f"Total Training time {training_time:.2f}")
    return W1, b1, W2, b2, W3, b3

def test(X_test, y_test, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3_test = foward_propagation(X_test, W1, b1, W2, b2, W3, b3)

    predictions = np.argmax(A3_test, axis = 1)

    accuracy = np.mean(predictions == y_test) *100
    print(f"Model Accuracy:  {accuracy: .2f}%")


W1 = 0.01*(np.random.randn(784,200))
b1 = np.zeros((1, 200))

W2 =  0.01*(np.random.randn(200,50))
b2 = np.zeros((1,50))

W3 =  0.01*(np.random.randn(50, 10))
b3 = np.zeros((1,10))

alpha = 0.1
epochs = 10
batch_size = 128
Y_train = one_hot(y_train)
X_train = x_train/255
X_test = x_test/255

W1, b1, W2, b2, W3, b3 = train(X_train, Y_train, y_train, W1, b1, W2, b2, W3, b3, alpha, epochs, batch_size)
test(X_test, y_test, W1, b1, W2, b2, W3, b3)



import numpy as np

class Node:
    def __init__(self, val, parents=None):
        self.val = val          
        self.parents = parents if parents else []  
        self.grad = 0

    def backward(self):
        for parent in self.parents:
            parent.grad += self.grad * self.local_grad(parent)
            parent.backward()

    def local_grad(self, parent):
        raise NotImplementedError

def multiply(x, y):
    result = x.val * y.val
    node = Node(result, parents=[x, y])

    def local_grad(parent):
        if parent == x:
            return y.val
        if parent == y:
            return x.val
        return 0

    node.local_grad = local_grad
    return node

def sin(x):
    result = np.sin(x.val)
    node = Node(result, parents=[x])

    def local_grad(parent):
        return np.cos(x.val)

    node.local_grad = local_grad
    return node

def square(x):
    result = x.val ** 2
    node = Node(result, parents=[x])

    def local_grad(parent):
        return 2 * x.val
    node.local_grad = local_grad
    return node

def cos(x):
    result = np.cos(x.val)
    node = Node(result, parents=[x])

    def local_grad(parent):
        return -np.sin(x.val)

    node.local_grad = local_grad
    return node

def add(x, y):
    result = x.val + y.val
    node = Node(result, parents=[x, y])

    def local_grad(parent):
        return 1

    node.local_grad = local_grad
    return node

def reciprocal(x):
    result = 1 / x.val
    node = Node(result, parents=[x])

    def local_grad(parent):
        return -1 / (x.val ** 2)

    node.local_grad = local_grad
    return node

if __name__ == "__main__":
    x1 = Node(1.0)
    w1 = Node(np.pi / 2)
    x2 = Node(2.0)
    w2 = Node(4.0)

    #x1 w1
    mul1 = multiply(x1, w1)       
    sin1 = sin(mul1) 
    sin_squared = square(sin1)

    #x2 w2
    mul2 = multiply(x2, w2)          
    cos2 = cos(mul2)       

    #full function
    sum_node = add(add(Node(2.0), sin_squared), cos2)  
    output = reciprocal(sum_node)

    #Gradients
    output.grad = 1
    output.backward()

    print("Gradient x1:", x1.grad) 
    print("Gradient w1:", w1.grad) 
    print("Gradient x2:", x2.grad) 
    print("Gradient w2:", w2.grad)  
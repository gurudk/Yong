import numpy as np


class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x,y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
        # f = self.creator
        # if f is not None:
        #     x = f.input
        #     x.grad = f.backward(self.grad)
        #     x.backward()




class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output

        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self,x):
        return x**2

    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx


class Exp(Function):
    def forward(self,x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx




data = np.array(1.0)
x = Variable(data)
print(x.data)

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)

A=Square()
B=Exp()
C=Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x



print(y.data)

y.grad = 1.0
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(x.grad)

y.grad = np.array(1.0)

C = y.creator
b = C.input
b.grad = C.backward(y.grad)
print(b.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

print(a.grad)

y.grad = np.array(1.0)
y.backward()
print(x.grad)





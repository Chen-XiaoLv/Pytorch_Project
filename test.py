import math
import random


def sigmoid(x):
    return math.tanh(x)


def ReLU(x):
    return x if x > 0 else 0


def derived_sigmiod(x):
    return x - x ** 2


def getRandom(a, b):
    return (b - a) * random.random() + a


def makeMatrix(m, n, val=0.0):
    return [[val] * n for _ in range(m)]


class BPNN:
    def __init__(self, n_in, n_out, n_hidden=10, lr=0.1, m=0.1):
        self.n_in = n_in + 1
        self.n_hidden = n_hidden + 1
        self.n_out = n_out
        self.lr = lr
        self.m = m
        self.weight_hidden = makeMatrix(self.n_in, self.n_hidden)
        self.weight_out = makeMatrix(self.n_hidden, self.n_out)

        for i, row in enumerate(self.weight_hidden):
            for j, val in enumerate(row):
                self.weight_hidden[i][j] = getRandom(-0.2, 0.2)

        for i, row in enumerate(self.weight_out):
            for j, val in enumerate(row):
                self.weight_out[i][j] = getRandom(-0.2, 0.2)

        self.in_matrix = [1.0] * self.n_in
        self.hidden_matrix = [1.0] * self.n_hidden
        self.out_matrix = [1.0] * self.n_out

        self.ci = makeMatrix(self.n_in, self.n_hidden)
        self.co = makeMatrix(self.n_hidden, self.n_out)

    def update(self, inputs):

        self.in_matrix[:-1] = inputs

        for i in range(self.n_hidden - 1):
            accumulate = 0
            for j in range(self.n_in - 1):
                accumulate += self.in_matrix[j] * self.weight_hidden[j][i]
            self.hidden_matrix[i] = sigmoid(accumulate)

        for i in range(self.n_out):
            accumulate = 0
            for j in range(self.n_hidden - 1):
                accumulate += self.hidden_matrix[j] * self.weight_out[j][i]
            self.out_matrix[i] = sigmoid(accumulate)
        return self.out_matrix[:]

    def backpropagate(self, target):

        out_err = [derived_sigmiod(o := self.out_matrix[i]) * (t - o) for i, t in enumerate(target)]

        hidden_err = [
            derived_sigmiod(self.hidden_matrix[i]) * sum(out_err[j] * self.weight_out[i][j] for j in range(self.n_out))
            for i in range(self.n_hidden)]

        for i in range(self.n_hidden):
            for j in range(self.n_out):
                change = self.hidden_matrix[i] * out_err[j]
                self.weight_out[i][j] += self.lr * change + self.m * self.co[i][j]
                self.co[i][j] = change

        for i in range(self.n_in):
            for j in range(self.n_hidden):
                change = hidden_err[j] * self.in_matrix[i]
                self.weight_hidden[i][j] += self.lr * change + self.m * self.ci[i][j]
                self.ci[i][j] = change

        err = 0.0
        for i, v in enumerate(target):
            err += (v - self.out_matrix[i]) ** 2
        err /= len(target)
        return math.sqrt(err)

    def train(self,data,epochs=1000):
        best_err=1e10
        for i in range(epochs):
            err=0.0
            for j in data:
                x=j[0]
                y=j[1]

                self.update(x)
                err+=self.backpropagate(y)
            if err<best_err:
                best_err=err
        print(best_err)

    def fit(self,x):
        return [self.update(i) for i in x]

# 固定模式
d=[
    [[1,0,1,0,1],[1]],
    [[1,0,1,1,1],[1]],
    [[1,1,1,0,1],[1]],
    [[1,0,1,1,1],[0]],
    [[1,1,1,1,1],[1]],
    [[1,0,1,1,1],[0]],
]
c=[
    [1,0,1,0,1],
    [1,0,1,0,1],
    [1,0,1,1,1],
    [1,0,1,0,1],
    [1,0,1,1,1],
    [1,0,1,0,1],
    [1,0,1,0,1],
    [1,0,1,1,1],
    [1,0,1,0,1],
    [1,1,1,0,1],
]

net=BPNN(5,1)

net.train(d)
print(net.fit(c))


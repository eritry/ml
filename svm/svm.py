import random
import numpy as np

class SVM:
    def __init__(self, kernel, C):
        self.X = []
        self.y = []
        self.weights = []
        self.b = 0.0
        self.C = C * 1.0
        self.kernel = kernel
        self.eps = 1e-4
        self.tol = 1e-1

    def calc(self, x):
        s = 0
        for j in range(len(self.weights)):
            s += self.y[j] * self.weights[j] * self.kernel(x, self.X[j])
        return np.sign(s + self.b)

    def doStep(self, i1, i2):
        if i1 == i2: return 0;
        a1 = self.weights[i1]
        a2 = self.weights[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]

        if y1 != y2:
            L = max(0.0, a2 - a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0.0, a2 + a1 - self.C)
            H = min(self.C, a2 + a1)

        if L == H: return 0

        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = k11 + k22 - 2 * k12

        E1 = self.calc(self.X[i1]) - y1
        E2 = self.calc(self.X[i2]) - y2
        if eta > 0:
            a2 = a2 + y2 * (E1 - E2) / eta
            if a2 < L: a2 = L
            elif a2 > H: a2 = H
        else: return 0

        if abs(self.weights[i2] - a2) < self.eps * (a2 + self.weights[i2] + self.eps):
            return 0

        s = y1 * y2
        a1 = a1 + s * (self.weights[i2] - a2)

        b1 = -E1 - y1 * (a1 - self.weights[i1]) * k11 - y2 * (a2 - self.weights[i2]) * k12 + self.b
        b2 = -E2 - y1 * (a1 - self.weights[i1]) * k12 - y2 * (a2 - self.weights[i2]) * k22 + self.b

        if a1 != 0 and a1 != self.C: self.b = b1
        elif a2 != 0 and a2 != self.C: self.b = b2
        else: self.b = (b1 + b2) / 2

        self.weights[i1] = a1
        self.weights[i2] = a2

    def examine(self, i2):
        y2 = self.y[i2]
        a2 = self.weights[i2]

        E2 = self.calc(self.X[i2]) - y2
        r2 = E2 * y2

        if (r2 < self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0):
            inds = list(range(len(self.weights)))
            random.shuffle(inds)

            for i in inds:
                a = self.weights[i]
                if a == 0 or a == self.C: continue
                if self.doStep(i, i2): return 1

            random.shuffle(inds)
            for i in inds:
                if self.doStep(i, i2): return 1
        return 0

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.weights = [0.0] * len(self.y)

        changed = 0
        examined = True
        iterations = 0
        while iterations < 50 and (changed > 0 or examined):
            changed = 0
            iterations += 1
            if examined:
                for i in range(len(self.weights)):
                    changed += self.examine(i)
            else:
                for i in range(len(self.weights)):
                    a = self.weights[i]
                    if a == 0 or a == self.C: continue
                    changed += self.examine(i)

            if examined: examined = False
            elif changed == 0: examined = True

    def predict(self, X_test):
        prediction = []
        for test in X_test:
            prediction.append(self.calc(test))
        return np.array(prediction)




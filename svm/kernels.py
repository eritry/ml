import numpy as np

def make_gaussian(b):
    def gaussian(x1, x2):
        return np.exp(-b * np.linalg.norm(x1 - x2))
    return gaussian

def make_linear(b):
    def linear(x1, x2):
        return x1.dot(x2)
    return linear

def make_polynom(b):
    def polynom(x1, x2):
        return np.power(1 + x1.dot(x2), b)
    return polynom



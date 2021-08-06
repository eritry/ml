from dtree import DT
import numpy as np
import math
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

class RF:
    def __init__(self, n_trees=3):
        self.forest = []
        self.indices = []
        self.n_trees = n_trees
    
    def fit(self, X, y):
        N = len(X)
        n = math.ceil(math.sqrt(len(X[0])))
        for i in range(self.n_trees):
            random_indices = np.random.choice(N, N, replace=True)
            
            nX = X[random_indices]
            ny = y[random_indices]

            self.forest.append(DT(None, n, stop=2))
            self.forest[-1].fit(nX, ny)
    
    def predict(self, X):
        result = []
        prediction = [tree.predict(X) for tree in self.forest]
        return stats.mode(prediction, axis=0)[0]
    
            
            
            
        
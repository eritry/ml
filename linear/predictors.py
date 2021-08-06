import numpy as np
import metrics

class LSM:
    def __init__(self, t):
        self.X = []
        self.y = []
        self.t = t
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.w = self.calculate(self.X, self.y).dot(self.y)

    def calculate(self, a, b):
        u, d, v = np.linalg.svd(a)

        dp = np.zeros((a.shape[0], a.shape[1]))
        dp[:d.shape[0], :d.shape[0]] = np.diag(d)

        r = (dp.T.dot(dp) + self.t * np.eye(dp.shape[1]))
        r = np.linalg.inv(r).dot(dp.T)
        
        return v.T.dot(r).dot(u.T)
        
    def predict(self, X):
        return np.dot(X, self.w)
    
    
class GD:
    def __init__(self, t=1, epoch=150, batch=8):
        self.X = []
        self.y = []
        self.epoch = epoch
        self.batch = batch
        self.smape_train = []
        self.smape_test = []
        self.w = []
        self.t = t

    def calculate_gradient(self, x_batch, y_batch, pred_batch):
        plus = np.abs(y_batch) + np.abs(pred_batch)
        minus = pred_batch - y_batch
        dLdy = minus / (plus * np.abs(minus) - pred_batch * np.abs(minus) / (np.abs(pred_batch) * np.square(plus)))
        
        #print(x_batch.shape, dLdy.shape)
        return np.dot(dLdy, x_batch) / len(x_batch)
        
    def fit(self, X, y, X_test = None, y_test = None):
        self.X = X
        self.y = y
        
        self.w = (np.random.rand(self.X.shape[1]) * 2 - 1) / 20
        for e in range(self.epoch):
            self.alpha = 1 / (e + 10)
            p = np.random.permutation(len(self.X))
            for i in range(0, len(self.X), self.batch):
                indices = p[i:i + self.batch]
                x_batch = self.X[indices]
                y_batch = self.y[indices]
                
                gradient = self.calculate_gradient(x_batch, y_batch, self.predict(x_batch))
                self.w = self.w * (1 - self.alpha * self.t) - self.alpha * (gradient + self.t * self.w)
            self.smape_train.append(metrics.smape(self.y, self.predict(self.X)))
            if y_test is not None: self.smape_test.append(metrics.smape(y_test, self.predict(X_test)))
                
            #if ((e + 1) % 10 == 0): print("Epoch:", str(e + 1), "smape:", metrics.smape(y, self.predict(self.X)))
                
    def predict(self, X):
        return np.dot(X, self.w)
   
    
class GA:
    def __init__(self, n_features, size=400, epochs=400):
        self.X = []
        self.y = []
        self.best = []
        self.size = size
        self.epochs = epochs
        self.n_features = n_features
        
    def fitness(self, solution):
        return 1/len(self.X)*np.sum(np.power(self.y-np.concatenate((self.X,np.ones((self.X.shape[0],1))),axis=1).dot(solution.reshape(-1,1)),2))

    def generate_population(self, size):
        solutions = ((np.random.rand(size, self.n_features + 1) * 2) - 1)
        return solutions

    def select_best(self, solutions):
        fitnesses = np.apply_along_axis(self.fitness, 1, solutions)
        return solutions[np.argsort(fitnesses, axis=0)[:20],:]

    def crossover(self, x, y):
        l = len(x) // 2
        crossed_x = x[:l].copy()
        crossed_y = y[:l].copy()
        y[:l], x[:l] = crossed_x, crossed_y
        return np.vstack((x, y))

    def mutation(self, solution):
        mutation_probability = 0.15
        if np.random.rand() < mutation_probability:
            solution = solution + (((np.random.rand(self.n_features + 1) * 2) - 1) * 0.1)
        return solution
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        solutions = self.generate_population(self.size)
        self.best = []

        for i in range(self.epochs):
            best_solutions = self.select_best(solutions)
            if i != 0 and self.fitness(best_solutions[0]) < self.fitness(self.best): self.best = best_solutions[0]
            else: self.best = best_solutions[0]
            
            new_population = np.array(self.best)
            for j in range(len(best_solutions) - 1):
                new_population = np.vstack((new_population, self.crossover(best_solutions[j], best_solutions[j + 1])))
            new_population = np.apply_along_axis(self.mutation, 1, new_population)
            solutions = new_population
        
    def predict(self, X):
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1).dot(self.best.transpose())

      
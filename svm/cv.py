import numpy as np

def get_partition(arr, k, onehot=False):
    if onehot: arr = np.argmax(arr, axis = 1)
        
    a = [(arr[i], i) for i in range(len(arr))]
    a = sorted(a, key=lambda t: t[0])

    ind = 0
    partition = [[] for i in range(k)]

    prev = a[0][0]

    i = 0
    while i < len(a):
        cur_ind = ind
        while i < len(a) and prev == a[i][0]:
            partition[cur_ind].append(a[i][1])
            i += 1
            cur_ind += 1
            if cur_ind == k: cur_ind = 0
        ind = cur_ind
        if i < len(a): prev = a[i][0]
    
    return [np.array(arr) for arr in partition]


def make_confusion_matrix(y_pred, y_true, num_of_classes, onehot=False):
    if onehot: y_true = np.argmax(y_true, axis = 1)
    matrix = [[0] * num_of_classes for i in range(num_of_classes)]
    
    for i in range(len(y_pred)):
        matrix[y_pred[i]][y_true[i]] += 1
        
    return matrix

def cross_validation(X, y, predictor, metric, k = 5, onehot=False):
    partition = get_partition(y, k, onehot)
    X_blocks = []
    y_blocks = []
    
    num_of_classes = len(np.unique(y, axis = 0))
    
    for part in partition:
        X_blocks.append(X[part])
        y_blocks.append(y[part])

    predictions = []
    y_true = []
    for i in range(k):
        X_valid, y_valid = X_blocks[i], y_blocks[i]
        X_train, y_train = np.concatenate(X_blocks[:i] + X_blocks[i + 1:]), np.concatenate(y_blocks[:i] + y_blocks[i + 1:])
        
        predictor.fit(X_train, y_train)
        prediction = predictor.predict(X_valid)

        predictions.append(prediction)
        y_true.append(y_valid)

    return metric(np.concatenate(predictions), np.concatenate(y_true))
        
    
    
    



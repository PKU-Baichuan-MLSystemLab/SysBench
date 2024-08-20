import numpy as np

def weighted_moving_average(x, y, window_size=5):
    y_smooth = np.zeros_like(y)
    n = len(x)
    
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)

        x_neigh = x[start:end]
        y_neigh = y[start:end]
        
        distances = np.abs(x_neigh - x[i])
        weights = 1 / (distances + 0.1)
        weights /= weights.sum()

        y_smooth[i] = np.dot(y_neigh, weights)
    
    return y_smooth
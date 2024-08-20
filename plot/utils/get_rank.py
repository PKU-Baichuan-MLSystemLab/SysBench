import numpy as np

def rank_columns_desc(data):
    n = data.shape[0]
    
    sorted_indices = np.argsort(-data, axis=0)
    
    ranks = np.empty_like(sorted_indices, dtype=int)
    
    for i in range(data.shape[1]):
        column = data[:, i]
        sorted_column = -column[sorted_indices[:, i]]

        _, first_index_positions = np.unique(sorted_column, return_index=True)
        
        first_index_positions = first_index_positions.tolist() + [n]
        # print(first_index_positions)
        
        ranks_for_uniques = np.zeros(n, dtype=float)
        last_idx = 0
        for j in range(len(first_index_positions) - 1):
            ranks_for_uniques[last_idx:first_index_positions[j+1]] = first_index_positions[j]
            last_idx = first_index_positions[j+1]
        
        # print(ranks_for_uniques)
        ranks[:, i] = ranks_for_uniques[np.argsort(sorted_indices[:, i])]
    
    return ranks

if __name__ == '__main__':
    data = np.array([
        [20, 2, 20],
        [10, 15, 5],
        [50, 25, 5],
        [30, 10, 20]
    ])
    
    # Compute ranks
    ranked_data = rank_columns_desc(data)
    print("Original Data:\n", data)
    print("Ranked Data:\n", ranked_data)
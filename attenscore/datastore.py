import os
import numpy as np
import torch

class DataStore:
    def __init__(self) -> None:
        self._data_store = {}
        self._split_indices = []
        self.has_extra = False
    
    def add_split_index(self, split_index : int, extra : bool = False) -> None:
        if extra:
            assert len(self._split_indices) == 0, "Extra split index can only be added at the beginning"
            self.has_extra = True
        self._split_indices.append(split_index)
    
    def append(self, layer_num : int, data : torch.Tensor) -> None:
        if data.dim() == 4:
            data = data.squeeze(0)
        # [np, sq, sk] -> [sq, sk]
        # print(f"Data shape: {data.shape}", ", indices:", self._split_indices)
        data = data.sum(dim=0, dtype=torch.float32)
        # print(f"Data shape: {data.shape}", data[0])
        data = data.cumsum(dim=-1).cpu().numpy()
        
        to_save_data = np.zeros((data.shape[0], 12 if self.has_extra else 11), dtype=np.float32)
        sliced_data = data[:, self._split_indices]
        to_save_data[:, :sliced_data.shape[1]] = sliced_data
        to_save_data[:, sliced_data.shape[1]] = data[:, -1]
        
        if layer_num not in self._data_store:
            self._data_store[layer_num] = []
        self._data_store[layer_num].append(to_save_data)
        # print(f"Data appended to layer {layer_num}, shape is {to_save_data.shape}")
    
    def _collect(self, layer_num : int) -> np.ndarray:
        if layer_num not in self._data_store:
            return None
        return np.vstack(self._data_store[layer_num])
    
    def get_keys(self) -> list:
        return list(self._data_store.keys())
    
    def save_data(self, save_path : str, file_name : str = '') -> None:
        split_indices_np = np.array(self._split_indices)
        os.makedirs(save_path, exist_ok=True)
        
        for layer_num in self._data_store:
            data = self._collect(layer_num)
            to_save = {
                "data": data,
                "split_indices": split_indices_np
            }
            fn = f"layer_{layer_num}_{file_name}.npy" if file_name else f"layer_{layer_num}.npy"
            np.save(os.path.join(save_path, fn), to_save)
            print(f"File saved: {fn}, with {data.shape[0]} samples")
    
    def load_data(self, load_path : str) -> None:
        assert len(self._data_store) == 0, "Data store is not empty"
        for file in os.listdir(load_path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(load_path, file), allow_pickle=True).item()
                layer_num = int(file.split("_")[1].split(".")[0])
                self._data_store[layer_num] = data["data"]
                self._split_indices = data["split_indices"]
    
    def clear(self) -> None:
        self._data_store = {}
        self._split_indices = []
        self.has_extra = False
        
    def get_split_indices(self) -> list:
        return self._split_indices

data_store = DataStore()

def get_data_store() -> DataStore:
    global data_store
    return data_store
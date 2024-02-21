## Tokenizer training data

### 1M.txt, 1K.txt
These files are the training data for BBPE tokenizer, created on 8.2.2024. They replaced the original files that had no deterministic history (and from the beginning were meant to be just random set to try things). The original files and the original trained tokenizer can be found in the deprecated/tokenizer directory (server alfa).

These new sets are random subsets of 30M_slice dataset from ZINC15 with a deterministic origin. The subsetting procedure was performed in the notebooks/tokenizer_bbpe_train.ipynb notebook. 

```python 
import numpy as np
from pathlib import Path

np.random.seed(42)

data_path = "../data/datasets/ZINC15/30M_slice/30M.smi"
slice_save_dir = "../tokenizer/training_data"

def random_slice(size, data_path, slice_save_path):
    with open(data_path, 'r') as f:
        data = np.array(f.read().splitlines())
        choice = np.random.choice(data, size, replace=False)

    with open(slice_save_path, 'w') as f:
        for item in choice:
            f.write(item + " ")
    
random_slice(1000000, data_path, slice_save_dir + "/1M.txt")
random_slice(1000, data_path, slice_save_dir + "/1K.txt")
```


### 240k_selfies.txt
created in notebooks/selfies_analysis.ipynb, it should maybe be 
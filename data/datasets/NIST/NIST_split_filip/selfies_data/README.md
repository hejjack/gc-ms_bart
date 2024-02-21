## This dataset is created by the following steps:
1. Create some of the jsonl mfX SMILES datasets using the guidlines in their corresponding README.md (doesn' matter which you use, they are all the same for this purpose). 
2. Translate the SMILES json dataset to a SELFIES json dataset using the following script:

```python
import sys
sys.path.append("..")

import selfies as sf
import json
from tqdm.notebook import tqdm
from pathlib import Path
from bart_spektro.selfies_tokenizer import hardcode_build_selfies_tokenizer

# autoreload libraries
%load_ext autoreload
%autoreload 2


def count_file_lines(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    counter = 0
    f = file_path.open("r")
    for line in f:
        if line.strip():
            counter += 1
    f.close()
    return counter

def dummy_generator(from_n_onwards=0):
    i = from_n_onwards
    while True:
        yield i
        i += 1

sel_tokenizer = hardcode_build_selfies_tokenizer()


def smiles_dataset_to_selfies_dataset(smiles_dataset_path, selfies_dataset_save_path, sel_tokenizer, source_id, seq_len=200):

    with open(smiles_dataset_path, "r") as f_smi, open(selfies_dataset_save_path, "w") as f_sel:
        for i in tqdm(dummy_generator()):
            smi_line = f_smi.readline()
            if not smi_line:
                break
            smi_row = json.loads(smi_line)
            smi_row["mol_repr"] = sf.encoder(smi_row["mol_repr"])
            tokenized_selfie = [source_id] + sel_tokenizer.encode(smi_row["mol_repr"]) + [sel_tokenizer.eos_token_id]
            assert len(tokenized_selfie) < seq_len, f"selfie: {tokenized_selfie}, len: {len(tokenized_selfie)} is too long!"
            smi_row["labels"] =  tokenized_selfie
            sel_line = json.dumps(smi_row)
            f_sel.write(sel_line + "\n")

for datatype in ["train", "valid", "test"]:
    smiles_dataset_to_selfies_dataset(f"../data/datasets/NIST/NIST_split_filip/mf100/{datatype}.jsonl", 
                                    f"../data/datasets/NIST/NIST_split_filip/selfies_data/{datatype}.jsonl", 
                                    sel_tokenizer, 
                                    sel_tokenizer.encode("[nist]")[0])
```

### PREPROCESSING STATS
are the same as in the original dataset

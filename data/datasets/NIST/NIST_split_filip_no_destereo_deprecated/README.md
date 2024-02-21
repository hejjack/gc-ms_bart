# NIST split created by FILIP JOZEFOV's notebook Noteboks/data_preprocessing.ipynb
- NIST20 without unidentifiable spectra (~60k)

## This dataset is created by the following steps:
1. Create jsonl files by the function msp_file_to_jsonl from spectra_process_utils.py (for 'train' and then 'test' and 'valid'):

```python
from spectra_process_utils import msp_file_to_jsonl
from pathlib import Path

dataset_path = Path("data/datasets/NIST/NIST_split_filip")
dataset_type = "train"
source_token = "<nist>"
msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                tokenizer_path,
                source_token,
                path_jsonl=dataset_path / f"{dataset_type}_{source_token}.jsonl",
                keep_spectra=True
                )
```

### PREPROCESSING STATS
 - test.jsonl
    0 no smiles
    48 smiles too long
    1 spectra corrupted
    697 spectra w/ too high mz
    2267 spectra w/ too many peaks
    totally 3013 issues
    discarded 2693/26365 spectra
    LENGTH: 23870
 - valid 
    0 no smiles
    39 smiles too long
    1 spectra corrupted
    663 spectra w/ too high mz
    2205 spectra w/ too many peaks
    totally 2908 issues
    discarded 2623/26493 spectra
    LENGTH: 23870
 - train
    0 no smiles
    406 smiles too long
    14 spectra corrupted
    6004 spectra w/ too high mz
    20226 spectra w/ too many peaks
    totally 26650 issues
    discarded 24049/237455 spectra
    LENGTH: 213406


## SEL_* variant of this dataset (the SELFIES representation)
was created using this code snippet. It was needed to modify selfies constraints to encode all the crazy molecules from NIST


THIS HAS TO BE CORRECTED (padding out, masks out)
```python
import selfies as sf
import json
from tqdm.notebook import tqdm
from pathlib import Path
from bart_spektro.selfies_tokenizer import hardcode_build_selfies_tokenizer

def dummy_generator(from_n_onwards=0):
    i = from_n_onwards
    while True:
        yield i
        i += 1

sel_tokenizer = hardcode_build_selfies_tokenizer()

# allowing 3 bonded Iodine atoms
default_constraints = sf.get_semantic_constraints()
default_constraints["I"] = 5
default_constraints["Ti"] = 13
default_constraints["P"] = 6
default_constraints["P-1"] = 6
sf.set_semantic_constraints(default_constraints)

def smiles_dataset_to_selfies_dataset(smiles_dataset_path, selfies_dataset_save_path, sel_tokenizer, source_id, seq_len=200):

    with open(smiles_dataset_path, "r") as f_smi, open(selfies_dataset_save_path, "w") as f_sel:
        for i in tqdm(dummy_generator()):
            smi_line = f_smi.readline()
            if not smi_line:
                break
            smi_row = json.loads(smi_line)
            smi_row["smiles"] = sf.encoder(smi_row["smiles"])
            tokenized_selfie = [source_id] + sel_tokenizer.encode(smi_row["smiles"]) + [sel_tokenizer.eos_token_id]
            assert len(tokenized_selfie) < seq_len, f"selfie: {tokenized_selfie}, len: {len(tokenized_selfie)} is too long!"
            smi_row["labels"] =  tokenized_selfie + [-100] * (seq_len - len(tokenized_selfie))
            assert len(smi_row["labels"]) == seq_len, f"selfie: {tokenized_selfie}, len: {len(tokenized_selfie)} labels len is different from seqlen!"
            sel_line = json.dumps(smi_row)
            f_sel.write(sel_line + "\n")

for datatype in ["train", "valid", "test"]:
    smiles_dataset_to_selfies_dataset(f"data/datasets/NIST/NIST_split_filip/{datatype}.jsonl", 
                                    f"data/datasets/NIST/NIST_split_filip/sel_{datatype}.jsonl", 
                                    sel_tokenizer, 
                                    sel_tokenizer.encode("[nist]")[0])
```
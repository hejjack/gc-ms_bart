## This dataset is created by the following steps:
1. Create msp splits using my updated version of FILIP JOZEFOV's notebook data/nist_cleaning_splitting.ipynb
   - it dorps ~60k spectra that don't have some form of proper identifier (smiles, inchikey)
   - split 0.8:0.1:0.1
2. Create jsonl files by the function msp_file_to_jsonl from spectra_process_utils.py (for 'train' and then 'test' and 'valid'):

```python
from spectra_process_utils import msp_file_to_jsonl
from pathlib import Path
from train_bart import build_tokenizer

tokenizer_type = "mf100"
tokenizer_path = f"tokenizer/bbpe_tokenizer/bart_bbpe_tokenizer_1M_{tokenizer_type}.model"
tokenizer = build_tokenizer(tokenizer_path)


for dataset_type in ["train", "valid", "test"]:
    dataset_path = Path("data/datasets/NIST/NIST_split_filip")
    source_token = "<nist>"
    msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                    tokenizer,
                    source_token,
                    path_jsonl=dataset_path / tokenizer_type / f"{dataset_type}.jsonl",
                    keep_spectra=True
                    )
```

# PREPROCESSING STATS
 - train.jsonl
   0 no mol_repr
   363 mol_reprs too long
   0 spectra corrupted
   5873 spectra w/ too high mz
   2344 spectra w/ too many peaks
   totally 8580 issues
   discarded 7292/232025 spectra 
  LENGTH: 224733 

 - valid.jsonl
   0 no mol_repr
   55 mol_reprs too long
   0 spectra corrupted
   709 spectra w/ too high mz
   283 spectra w/ too many peaks
   totally 1047 issues
   discarded 876/29053 spectra 
  LENGTH: 28177

 - test.jsonl
   0 no mol_repr
   52 mol_reprs too long
   0 spectra corrupted
   780 spectra w/ too high mz
   302 spectra w/ too many peaks
   totally 1134 issues
   discarded 951/29218 spectra 
  LENGTH: 28267                          
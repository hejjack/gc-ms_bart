This dataset was created by filtering 30M ZINC slice dataset. The filtering process 
suits the RASSP model trained by Ales Krenek (following the official RASSP model repo)

The data went throuhg: CANONICALIZATION, DESTEREOCHEMICALIZATION, DEDUPLICATION, LONG SMILES FILTERING (100), CORRUPTED SMILES FILTERING (done on the original 30M ZINC slice dataset). There is NO FILTERING regarding max_num_of_peaks or max_mz or other.

This dataset is created by the following steps:
1. Get the 30M_rassp.smi (A.Krenek's filtering method used on a 30M ZINC slice dataset)
2. feed this to msp_preprocess_rassp.py in a following way:

```bash
python ../data/msp_preprocess_rassp.py --input-dir ../data/datasets/30M_rassp/rassp_gen/msps \
                               --output-dir ../data/datasets/30M_rassp/rassp_gen \
                               --config-file ../configs/preprocess_config_RASSP.yaml \
                               --num-processes 32 \
                               --concat \
                               --clean
```

with a config preprocess_config_RASSP.yaml:

```yaml
do_preprocess: False
train_split_ratio: 0.9
valid_split_ratio: 0.05
test_split_ratio: 0.05
neims_dir: "../../NEIMS"
seed: 42
```

There is 4.8M SMILES in the dataset. The preprocessing further cut the number of molecules,
finally the lengths of splits are:
    - train 4364744
    - valid 242486
    - test 242486 
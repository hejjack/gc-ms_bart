This dataset was created by filtering 30M ZINC slice dataset. The filtering process 
suits the RASSP model trained by Ales Krenek (following the official RASSP model repo)

The data went through: CANONICALIZATION, DESTEREOCHEMICALIZATION, DEDUPLICATION, LONG SMILES FILTERING (100), CORRUPTED SMILES FILTERING (done on the original 30M ZINC slice dataset). 
During the NEIMS generation, the data was further filtered based on `max_mz` and `max_peaks` 
The data consists of json dicts on each line containing "smiles", "mz" and "intensity". Further preprocessing is done on-the-fly.


This dataset is created by the following steps:
1. Get the 4_8M.smi (A.Krenek's filtering method used on a 30M ZINC slice dataset)
2. feed this to run_preprocess_neims.sh

```bash
echo $CONDA_PREFIX should be ~/miniconda3/envs/NEIMSpy3
OMP_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0  # sth for OMP to not throw INFOs
ID=4_8M
python ../data/smi_preprocess_neims.py \
                        --smiles-path ../data/datasets/${ID}/${ID}.smi \
                        --dataset-id $ID \
                        --num-workers 20 \
                        --config-file ../configs/preprocess_config_NEIMS.yaml \
                        --auto-remove \
```

3. Set a config to preprocess_config_NEIMS.yaml:
```yaml
tmp_dir: "../data/datasets/tmp"
log_dir: "../data/datasets/log"
output_dir: "../data/datasets/4_8M/neims_gen"
phases_to_perform: [1,2,3,4,5]   # 1_5, order is not important
lost_chunks: []  
data_has_header: False     # whether the input has clean smiles file or csv (with <smiles zinc_id> structure)
do_preprocess: False
max_smiles_len: 100
max_mz: 500
max_peaks: 300
log_base: 1.7
log_shift: 9
source_token: "<neims>"
train_split_ratio: 0.9
valid_split_ratio: 0.05
test_split_ratio: 0.05
neims_dir: "../../NEIMS"
seed: 42
```

## RAM overflow
Be careful about RAM overflow (500GB machine wasn't enough for the NEIMS generation in phase3). If some of the processes fail, you can use a combination of `lost_chunks` and `phases_to_perform` parameters to rerun only the failed chunks from a saved checkpoint. Remember to keep the `num-workers` parameter the same so the program finds all the right chunk checkpoints. 

In the end, there is 4.8M SMILES in the dataset. The preprocessing further cut the number of molecules,
finally the lengths of splits are:
    - train 4364716
    - valid 242490
    - test  242480
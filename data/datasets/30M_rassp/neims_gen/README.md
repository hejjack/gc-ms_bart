This dataset was created by filtering 30M ZINC slice dataset. The filtering process 
suits the RASSP model trained by Ales Krenek (following the official RASSP model repo)

This dataset is created by the following steps:
1. Get the 30M_rassp.smi (A.Krenek's filtering method used on a 30M ZINC slice dataset)
2. feed this to run_preprocess_neims.sh

```bash
echo $CONDA_PREFIX should be ~/miniconda3/envs/NEIMSpy3
OMP_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0  # sth for OMP to not throw INFOs
ID=30M_rassp
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
output_dir: "../data/datasets/30M_rassp/neims_gen"
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



In the end, there is 4.8M SMILES in the dataset. The preprocessing further cut the number of molecules,
finally the lengths of splits are:
    - train 428384 8 (aktuali)
    - valid 237990
    - test  237990
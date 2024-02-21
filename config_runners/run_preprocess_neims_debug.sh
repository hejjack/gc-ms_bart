echo $CONDA_PREFIX should be ~/miniconda3/envs/NEIMSpy3
OMP_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0  # sth for OMP to not throw INFOs
ID=1K
python ../data/smi_preprocess_neims.py \
                        --smiles-path ../data/datasets/${ID}/${ID}.smi \
                        --dataset-id $ID \
                        --num-workers 1 \
                        --config-file ../configs/preprocess_config_NEIMS.yaml \
                        --auto-remove \
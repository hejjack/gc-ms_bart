These datasets are meant for denovo evaluation of the BARTSpectro model. They are composed of the same data as the NIST_split_filip dataset, but with additional denovo information added to each query. The additional information is: "index_of_closest": "spectra_sim_of_closest", "smiles_sim_of_closest". The similarity is calculated using Morgan fingerprints and cosine similarity.

These datasets were composed by following run script:

```bash
SPLIT_NAME=test
python ../precompute_denovo_index.py \
           --reference ../data/datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../data/datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../data/datasets/NIST/NIST_split_filip/denovo_data/${SPLIT_NAME}_with_denovo_info.jsonl \
           --num_processes 32
```

It took 4:48:41 to run on 32 processes.

THOUGH!
In the newer script version this is equivalent to:
```bash
SPLIT_NAME=valid
FP_TYPE=morgan
SIMIL_FUN=cosine

python ../precompute_denovo_index.py \
           --reference ../data/datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../data/datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../data/datasets/NIST/NIST_split_filip/denovo_data/${SPLIT_NAME}_with_denovo_info.jsonl \
           --fingerprint_type ${FP_TYPE} \
           --simil_function ${SIMIL_FUN}
           --num_processes 32
```

Different fingerprint types and similarity functions can be efficiently computed using the same script. If there already is a precomputed denovo dataset, the script only add a column with the new selected pair of fp and simil. E.g., daylight fp and tanimoto similarity:

```bash
SPLIT_NAME=valid
FP_TYPE=daylight
SIMIL_FUN=tanimoto

python ../precompute_denovo_index.py \
           --reference ../data/datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../data/datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../data/datasets/NIST/NIST_split_filip/denovo_data/${SPLIT_NAME}_with_denovo_info.jsonl \
           --fingerprint_type ${FP_TYPE} \
           --simil_function ${SIMIL_FUN}
           --num_processes 32
```

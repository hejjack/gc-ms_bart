These datasets are meant for denovo evaluation of the BARTSpectro model. They are composed of the same data as the NIST_split_filip dataset, but with additional denovo information added to each query. The additional information is: "index_of_closest": "spectra_sim_of_closest", "smiles_sim_of_closest". The similarity is calculated using Morgan fingerprints and cosine similarity.

These datasets were composed by following run script:

```bash
SPLIT_NAME=test
python ../data/precompute_denovo_index.py \
           --reference ../data/datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../data/datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../data/datasets/NIST/NIST_split_filip/denovo_data/morgan_cosine/${SPLIT_NAME}_with_denovo_info.jsonl \
           --num_processes 32
```

It took 4:48:41 to run on 32 processes.

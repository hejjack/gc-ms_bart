SPLIT_NAME=test
python ../precompute_denovo_index.py \
           --reference ../datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../datasets/NIST/NIST_split_filip/denovo_data/${SPLIT_NAME}_with_denovo_info.jsonl \
           --num_processes 32 \

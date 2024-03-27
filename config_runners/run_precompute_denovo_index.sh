SPLIT_NAME=test
FP_TYPE=daylight
SIMIL_FUN=tanimoto

python ../precompute_denovo_index.py \
           --reference ../data/datasets/NIST/NIST_split_filip/train.jsonl \
           --query ../data/datasets/NIST/NIST_split_filip/${SPLIT_NAME}.jsonl \
           --outfile ../data/datasets/NIST/NIST_split_filip/denovo_data/${SPLIT_NAME}_with_denovo_info.jsonl \
           --num_processes 16 \
           --fingerprint_type ${FP_TYPE} \
           --simil_function ${SIMIL_FUN}
 in with

￼Masaryk University
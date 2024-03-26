# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST_denovo/1697808199_valid_0:9248_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --do-denovo

# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST_denovo/1697801918_valid_0:50_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --do-denovo

python ../evaluate_predictions.py --predictions-path ../predictions/lucky-tree-293_4_8M_rassp1_neims1_224kPretrain/NIST_denovo/1710756718_valid_full_beam10_peaks200/predictions.jsonl \
                                  --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
                                  --config-file ../configs/eval_config_previous_generation.yaml \
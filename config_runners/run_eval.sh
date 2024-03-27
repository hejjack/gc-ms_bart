# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST_denovo/1697808199_valid_0:9248_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --do-denovo

# python ../evaluate_predictions.py --predictions-path ../predictions/fresh-blaze-258_4_8M_rassp1_neims1_224kPretrain/NIST_denovo/1697801918_valid_0:50_beam10/predictions.jsonl \
#                                   --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
#                                   --do-denovo

python ../evaluate_predictions.py --predictions-path ../predictions/tough-jazz-406_rassp1_neims1_200pt_200ft/NIST_denovo/1711198593_valid_full_200peaks/predictions.jsonl \
                                  --labels-path ../data/datasets/NIST/NIST_split_filip/denovo_data/valid_with_denovo_info.jsonl \
                                  --config-file ../configs/eval_config.yaml \
python ../predict.py --checkpoint ../checkpoints/finetune_deprecated/lucky-tree-293_4_8M_rassp1_neims1_224kPretrain/checkpoint-73440 \
                                            --output-folder ../predictions \
                                            --config-file ../configs/predict_config_nist_valid_old_gen2.yaml \
                                            # --data-range "0:200"